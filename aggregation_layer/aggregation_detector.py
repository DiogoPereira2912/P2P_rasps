from aggregation_algs.algs import ALGS_DICT
from yaml import Loader, load
from client.mqtt_layer import Communication_Layer
import json, threading, time, uuid, base64, io, torch
from queue import Empty 
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore")

from aggregation_algs.aggregation_utils import resolve_targets_by_index

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
        self.server_ip = None # ip descoberto com o node_id = 0
        self.server_id = None # replace de . por _ para estar de acordo com a bridge

        self.current_peer_list = []
        self.min_peers = self.config["min_peers"]
        self.aggregation_dest_indices = self.config["routing_topology"]["aggregation_topology"]

        # path to get model
        self.update_model_path = "../models/base/updated_model.pt"

        self.last_ts = None
        self.last_round_start = 0
        self.started_training_time = None
        self.remote_weights = []

        if self.mode == "federated":
            if self.node_id == self.central_id:
                print("[AGGREGATOR] Eu sou o SERVIDOR CENTRAL (Main).")
                self._setup_mqtt_client(subscribe_topic="+/agg")
                self._start_agg_worker()
            else:
                print("[AGGREGATOR] Modo Federated: Sou um Worker.")
                pass 
        else: 
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

    def _start_agg_worker(self):
        '''
        Inicia o worker de agregação em uma thread separada.
        '''
        agg_thread = threading.Thread(target=self.agg_worker, daemon=True)
        agg_thread.start()

    def _deserialize_model(self, b64_weight_str):
            """Converte String Base64 -> PyTorch State Dict"""
            try:
                weight_bytes = base64.b64decode(b64_weight_str)
                buffer = io.BytesIO(weight_bytes)
                return torch.load(buffer, map_location='cpu')
            except Exception as e:
                print(f"❌ Erro ao deserializar modelo: {e}")
                return None

    def _serialize_model(self, state_dict):
        """Converte PyTorch State Dict -> String Base64"""
        try:
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"❌ Erro ao serializar modelo: {e}")
            return None

    def aggregate(self, models_state_dicts, method):
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
        return ALGS_DICT[method](models_state_dicts)

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
        weights = data["weights"]

        if node_id in buffer:
            existing_ts = buffer[node_id]["ts"]
            if msg_ts > existing_ts:
                buffer[node_id] = {'ts': msg_ts, 'weights': weights}
        else:
            buffer[node_id] = {'ts': msg_ts, 'weights': weights}

    def save_model(self, weights, path):
        '''
        Cria instancia vazia do modelo e carrega os pesos agregados antes de salvar.
        Args:
            weights (dict): Dicionário com os pesos do modelo.
            path (str): Caminho onde o modelo será salvo.
        '''
        temp_model = YOLO("yolov8n.yaml") 
        temp_model.model.load_state_dict(weights, strict=False)
        
        ckpt = {
            'model': temp_model.model,
            'epoch': -1,
            'train_args': {},
        }
        torch.save(ckpt, path)
        return True

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
                    else:
                        self._process_msg_into_buffer(data, current_round_buffer)
                    self.mqtt_com.msg_queue.task_done()
                except Empty:
                    continue
            print(f"🔒 [AGG] Janela fechada. Total de nós recolhidos: {len(current_round_buffer)}")

            if len(current_round_buffer) > 0:
                for node_id, data in current_round_buffer.items():
                    state_dict = self._deserialize_model(data['weights']) 
                    self.remote_weights.append(state_dict)
                aggregated_weights = self.aggregate(self.remote_weights, method="avg")
                agg_weights_b64 = self._serialize_model(aggregated_weights)
                payload = {
                    "id": self.broker_id,
                    "agg_weights": agg_weights_b64, 
                    "ts": time.time()
                }
                if self.mode == "federated":
                    self.mqtt_com.publish(payload, topic=f"{self.broker_id}/train")
                    print("Publicar em", f"{self.broker_id}/train")
                else:
                    targets = resolve_targets_by_index(self.current_peer_list, self.aggregation_dest_indices)
                    if not targets:
                        # IF THERE ARE NO SPECIFIC TARGETS, BROADCAST TO ALL
                        self.mqtt_com.publish(payload, topic=f"{self.broker_id}/train")
                    else:
                        for ip in targets:
                            target_id = ip.replace(".", "_")
                            self.mqtt_com.publish(payload, topic=f"{target_id}/train")
                self.save_model(aggregated_weights, self.update_model_path)
                self.remote_weights = []
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