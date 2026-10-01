from aggregation_algs.algs import ALGS_DICT
from yaml import Loader, load
from client.mqtt_layer import Communication_Layer
import json, threading, time, uuid, os
from queue import Empty 
import pandas as pd
import numpy as np
from deltalake.writer import write_deltalake
from aggregation_algs.aggregation_utils import resolve_targets_by_index, load_class_mappings

import warnings
warnings.filterwarnings("ignore")

WINDOW_DURATION = 5.0 # Duração da janela de agregação em segundos

class Aggregator:
 
    def __init__(self):

        with open("client/config.yaml", "r") as file:
            self.config = load(file, Loader=Loader)
 
        self.mosquitto_port = self.config["mosquitto_port"]
        self.broadcast_port = self.config["broadcast_port"]
        self.broadcast_mask = self.config["broadcast_mask"]
        self.peer_ip = self.config["peer_ip"]
        self.broker_id = self.peer_ip.replace(".", "_")
        self.node_id = self.config["node_id"]

        self.mode = self.config["mode"]
        self.central_id = self.config["central_id"]
        
        self.server_ip = None 
        self.server_id = None 
        
        self.is_dynamic_server = False

        self.current_peer_list = []
        self.min_peers = self.config["min_peers"]
        self.aggregation_dest_indices = self.config["routing_topology"]["aggregation_topology"]

        self.last_ts = None
        self.last_round_start = 0
        self.started_training_time = None
        self.remote_preds = {}
        self.pred_probs_dict = {}
        self.test_data = None
        self.phase = "IDLE"
        self.label_map, self.reverse_label_map, self.class_list = load_class_mappings("labeling/global_rules.yaml")
        
        with open("labeling/global_rules.yaml", "r") as file:
            self.global_rules = load(file, Loader=Loader)

        # --- MAPA DINÂMICO DE PRIORIDADES ---
        self.priority_map = {}
        for rule in self.global_rules.get('regras', []):
            label = rule['label']       
            prio = rule['prioridade']   
            idx = self.label_map.get(label)
            if idx is not None:
                self.priority_map[idx] = prio

        self.test_data_path = f"data_exports/local_inf_data_{self.config['device_id']}"

        # if self.mode == "federated":
        #     if self.node_id == self.central_id:
        #         print("[AGGREGATOR] Eu sou o SERVIDOR CENTRAL (Main).")
        #         self._setup_mqtt_client(subscribe_topic="+/agg")
        #         self._start_agg_worker()
        #     else:
        #         print("[AGGREGATOR] Modo Federated: Sou um Worker.")
        #         pass 
        # else: 
        #     self._setup_mqtt_client(subscribe_topic="+/agg")
        #     self._start_agg_worker()
 
        # Todos os nós iniciam a escuta MQTT e o Worker para poderem receber ordens dinâmicas
        self._setup_mqtt_client(subscribe_topic="+/agg")
        self._start_agg_worker()

    def _setup_mqtt_client(self, subscribe_topic):
        """
        Cria o cliente MQTT e faz o subscribe ao tópico
        """
        self.mqtt_com = Communication_Layer(
            broker=self.peer_ip,
            port=self.mosquitto_port,
            client_id=f"aggregation_{self.broker_id}_{str(uuid.uuid4())[:4]}",
            base_topic="",
            qos=1,
        )
        self.mqtt_com.subscribe(topic=subscribe_topic)
        self.mqtt_com.subscribe("system/peers")
        self.mqtt_com.subscribe("system/control/#") 
        self.mqtt_com.client.message_callback_add("system/control/#", self.on_control_message)

    def on_control_message(self, client, userdata, msg):
        """
        Muda o comportamento do Aggregator consoante a ordem do Streamlit
        """
        try:
            payload = json.loads(msg.payload.decode())
            cmd = payload.get("command")
            
            config_recebida = payload.get("config", {})
            
            if "mode" in config_recebida:
                self.mode = "federated" if config_recebida["mode"] == "FL" else "gossip"
                print(f"⚙️ MODO AGGREGATOR FORÇADO: {self.mode.upper()}")
                
            if "central_ip" in config_recebida:
                self.server_ip = config_recebida["central_ip"]
                self.server_id = self.server_ip.replace(".", "_")
                self.is_dynamic_server = True 
                
                if self.peer_ip == self.server_ip:
                    print("👑 Fui promovido a MAIN SERVER (Agregador)!")
                else:
                    print(f"👷 Sou WORKER (Agregador). O orquestrador é o {self.server_ip}")

            self.phase = cmd

        except Exception as e:
            print(f"Erro no controlo Aggregator: {e}")

    def _verify_central_server(self, peers_list):
        """
        Descobre quem é o servidor central na lista de peers.
        Se a UI já o definiu dinamicamente, ignora a leitura do YAML.
        """
        if self.is_dynamic_server:
            return
            
        if self.node_id == self.central_id:
            self.server_ip = self.peer_ip
            self.server_id = self.server_ip.replace(".", "_")
            print("[AGGREGATOR] MAIN SERVER")
        else:
            print("[AGGREGATOR] WORKER")
            for p in peers_list:
                if p[1] == self.central_id:
                    self.server_ip = p[0]
                    self.server_id = self.server_ip.replace(".", "_")

    def _start_agg_worker(self):
        '''
        Inicia o worker de agregação em uma thread separada.
        '''
        agg_thread = threading.Thread(target=self.agg_worker, daemon=True)
        agg_thread.start()

    def aggregate(self, params_dict, method, **kwargs):
        '''
        Agrega os parâmetros recebidos usando o método especificado.
        Args:
            params_dict (dict): Dicionário com os parâmetros dos nós.
            method (str): Método de agregação a ser usado.
        Returns:
            dict: Parâmetros agregados.
        '''
        if method not in ALGS_DICT:
            raise ValueError(f"Método de agregação '{method}' não suportado.")
        return ALGS_DICT[method](params_dict, **kwargs)

    def _process_msg_into_buffer(self, data, buffer):
        '''
        Processa uma mensagem recebida e atualiza o buffer temporal 
        da janela de agregação com os parâmetros mais recentes.

        Args:
            data (dict): Dados recebidos de um nó.
            buffer (dict): Buffer temporal para armazenar os parâmetros.
        '''
        if "id" not in data or "ts" not in data:
            return

        msg_ts = data["ts"]
        node_id = data["id"]
        test_data = data["data"]
        pred_probs = data["pred_probs"]
        preds = data["preds"]

        if node_id in buffer:
            existing_ts = buffer[node_id]["ts"]
            if msg_ts > existing_ts:
                buffer[node_id] = {'ts': msg_ts,'data': test_data, 'pred_probs': pred_probs, 'preds': preds}
        else:
            buffer[node_id] = {'ts': msg_ts, 'data': test_data, 'pred_probs': pred_probs, 'preds': preds}

    def agg_worker(self):
        '''
        Worker que recolhe os parâmetros treinados dos nós e agrega-os periodicamente.
        1. Espera pela primeira mensagem para iniciar a janela de recolha.
        2. Abre uma janela de tempo (5 segundos) para recolher mensagens.
        3. Após o término da janela, agrega os parâmetros recebidos.
        4. Publica os parâmetros agregados para os destinos definidos.
        '''
        last_processed_ts = {} 

        while True:
            topic, first_data = self.mqtt_com.msg_queue.get()
            
            if topic == "system/peers":
                self.current_peer_list = first_data
                print("PEERS CONHECIDOS:", len(self.current_peer_list))
                print("PEERS NECESSÁRIOS:", self.min_peers)
                self.mqtt_com.msg_queue.task_done()
                continue

            if len(self.current_peer_list) < self.min_peers: 
                self.mqtt_com.msg_queue.task_done()
                continue

            if "id" not in first_data or "ts" not in first_data:
                self.mqtt_com.msg_queue.task_done()
                continue

            node_id = first_data["id"]
            msg_ts = first_data["ts"]

            if node_id in last_processed_ts and msg_ts <= last_processed_ts[node_id]:
                self.mqtt_com.msg_queue.task_done()
                continue 

            last_processed_ts[node_id] = msg_ts

            print(f"⏳ [AGG] Recebi dados NOVOS. A abrir janela de {WINDOW_DURATION}s...")
            print("PHASE ATUAL:", self.phase)
            print(f"[{self.broker_id}] RECEIVED on {topic}: {first_data}")
            
            collection_start_time = time.time()
            current_round_buffer = {}
            
            self._process_msg_into_buffer(first_data, current_round_buffer)
            self.mqtt_com.msg_queue.task_done()

            while (time.time() - collection_start_time) < WINDOW_DURATION:
                try:
                    topic, data = self.mqtt_com.msg_queue.get(timeout=0.5)
                    if topic == "system/peers":
                        self.current_peer_list = data
                        if self.mode == "federated":
                            self._verify_central_server(self.current_peer_list)
                    else:
                        self._process_msg_into_buffer(data, current_round_buffer)
                    self.mqtt_com.msg_queue.task_done()
                except Empty:
                    continue
            print(f"🔒 [AGG] Janela fechada. Total de nós recolhidos: {len(current_round_buffer)}")

            if len(current_round_buffer) > 0:

                self.remote_preds = {k: v['preds'] for k, v in current_round_buffer.items()}
                self.pred_probs_dict = {k: v['pred_probs'] for k, v in current_round_buffer.items()}
                #final_probs = self.aggregate(self.pred_probs_dict, method="avg")
                final_probs = self.aggregate(
                    self.pred_probs_dict, 
                    method="fallback",
                    threshold=0.3,              
                    fallback_mode="priority", 
                    priority_map=self.priority_map
                )

                # sacar idx e label correspondente para a classe mais provável
                final_prob_idx = np.argmax(final_probs[0]) if final_probs else None
                final_label = self.reverse_label_map.get(final_prob_idx, "UNKNOWN")

                if self.phase in ["INFERENCE", "METRICS"]:
                    print(f"💾 [IO] A tentar gravar. Procuro pelo ID: {self.node_id}")
                    if self.node_id in current_round_buffer:

                        my_data = current_round_buffer[self.node_id]

                        raw_data = my_data['data']
                        if isinstance(raw_data, dict):
                            raw_data = [raw_data] 
                        full_test_df = pd.DataFrame(raw_data)
                        
                        full_test_df["local_probs"] = [my_data['pred_probs']]
                        full_test_df['local_pred'] = [self.reverse_label_map.get(x, "UNKNOWN") for x in my_data['preds']]
                        full_test_df['global_probs'] = [final_probs]
                        full_test_df['global_pred'] = final_label

                        if not os.path.exists(self.test_data_path):
                            os.makedirs(self.test_data_path)
                            
                        write_deltalake(self.test_data_path, full_test_df, mode="append")
                        print(f"✅ [SUCESSO] Dados gravados em {self.test_data_path}")

                # only central server publishes the aggregated prediction to the workers in federated mode
                if self.mode == "federated" and self.peer_ip != self.server_ip:
                    self.remote_preds = {}
                    self.pred_probs_dict = {}
                    continue 

                payload = {
                    "id": self.broker_id,
                    "ts": time.time(),
                    "final_pred": final_label, 
                }
                print(f"Final aggregated prediction: {final_label} with probs {final_probs}")
                if self.mode == "federated":
                    self.mqtt_com.publish(payload, topic=f"{self.broker_id}/train")
                    print("Publicar em", f"{self.broker_id}/train")
                else:
                    targets = resolve_targets_by_index(self.current_peer_list, self.aggregation_dest_indices)
                    if not targets:
                        self.mqtt_com.publish(payload, topic=f"{self.broker_id}/train")
                    else:
                        for ip in targets:
                            target_id = ip.replace(".", "_")
                            self.mqtt_com.publish(payload, topic=f"{target_id}/train")
                self.remote_preds = {}
                self.pred_probs_dict = {}
            else:
                print("⚠️ [AGG] Janela fechou sem dados válidos.")
                continue

if __name__ == "__main__":
    aggregator = Aggregator()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass