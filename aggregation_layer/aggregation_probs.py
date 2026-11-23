from aggregation_algs.algs import ALGS_DICT
from yaml import Loader, load
from client.mqtt_layer import Communication_Layer
import json, threading, time, uuid, os
from queue import Empty 
import pandas as pd
import numpy as np
from deltalake.writer import write_deltalake
from aggregation_algs.aggregation_utils import resolve_targets_by_index, load_class_mappings
from prometheus_client import start_http_server, Counter, Gauge, Histogram

import warnings 
warnings.filterwarnings("ignore")

WINDOW_DURATION = 5.0 # Duração da janela de agregação em segundos
PROMETHEUS_PORT = 8002  # 8000 = pipeline, 8001 = video server, 8080 = cAdvisor

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
        self.device_id = self.config["device_id"]

        # Método de consenso: "rules" (mesmo critério do ground truth, global_rules.yaml)
        # ou "avg"/"fallback" (baselines). Cada linha gravada guarda o método usado.
        self.agg_method = self.config.get("aggregation_method", "rules")
        # Identificador da experiência (enviado pela UI no comando INFERENCE)
        self.run_id = None
        print(f"⚙️ Método de agregação: {self.agg_method}")

        self._setup_prometheus_metrics()

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

            if cmd == "INFERENCE":
                self.run_id = config_recebida.get("run_id") or time.strftime("%Y%m%d_%H%M%S")
                print(f"🏷️  Run: {self.run_id}")

            self.phase = cmd

        except Exception as e:
            print(f"Erro no controlo Aggregator: {e}")

    def _setup_prometheus_metrics(self):
        """
        Métricas para o Grafana (dashboard "Classificação & Consenso").
        A latência entre nós só é válida com relógios sincronizados (tools/sync_clocks.py).
        """
        labels = ["node", "mode", "method"]
        self.m_rounds = Counter("aidi_agg_rounds_total", "Rondas de consenso", labels)
        self.m_decisions = Counter("aidi_agg_decisions_total", "Decisões globais por classe", labels + ["decision"])
        self.m_divergence = Counter("aidi_agg_divergence_total", "Rondas em que a decisão global difere da local", labels)
        self.m_nodes = Gauge("aidi_agg_nodes_in_round", "Nº de nós na última ronda", ["node"])
        self.m_msg_delay = Histogram(
            "aidi_agg_message_delay_seconds", "Predição publicada -> recebida no agregador",
            ["node", "mode"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
        )
        self.m_decision_delay = Histogram(
            "aidi_agg_decision_delay_seconds", "Predição publicada -> decisão global (inclui a janela)",
            ["node", "mode"], buckets=(0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15),
        )
        try:
            start_http_server(PROMETHEUS_PORT)
            print(f"📡 Prometheus exporter (aggregation) na porta {PROMETHEUS_PORT}")
        except Exception as e:
            print(f"⚠ Aviso: Prometheus exporter não arrancou: {e}")

    def _run_aggregation(self, probs_dict):
        """Aplica o método configurado, com os parâmetros que cada um precisa."""
        if self.agg_method == "rules":
            return self.aggregate(probs_dict, method="rules",
                                  rules_config=self.global_rules, label_map=self.label_map)
        if self.agg_method == "fallback":
            return self.aggregate(probs_dict, method="fallback", threshold=0.3,
                                  fallback_mode="priority", priority_map=self.priority_map)
        return self.aggregate(probs_dict, method=self.agg_method)

    def _save_round(self, buffer, node_ids, final_label, final_probs, decision_ts):
        """
        Grava em local_inf_data_<nó> uma linha por amostra dos nós em node_ids, com tudo
        o que é preciso para a análise offline (analysis/):
          - device/node_id: nó que fez a predição local
          - mode, agg_method, run_id: para separar GL vs FL e métodos nos gráficos
          - pred_ts, decision_ts: latência predição -> decisão
          - round_probs: probabilidades de TODOS os nós da ronda (permite recalcular
            offline qualquer método de consenso sobre exatamente os mesmos dados)
        """
        global_vector = [float(p) for p in np.asarray(final_probs)[0]]
        round_probs = json.dumps({str(k): [float(p) for p in v['pred_probs'][0]] for k, v in buffer.items()})
        frames = []
        for nid in node_ids:
            entry = buffer[nid]
            raw_data = entry['data']
            if isinstance(raw_data, dict):
                raw_data = [raw_data]
            df = pd.DataFrame(raw_data)
            n_rows = len(df)
            df["device"] = entry.get("device") or str(nid)
            df["node_id"] = nid
            df["local_probs"] = [[float(p) for p in row] for row in entry['pred_probs']]
            df["local_pred"] = [self.reverse_label_map.get(x, "UNKNOWN") for x in entry['preds']]
            df["global_probs"] = [global_vector] * n_rows
            df["global_pred"] = final_label
            df["mode"] = self.mode
            df["agg_method"] = self.agg_method
            df["run_id"] = self.run_id or "sem_run_id"
            df["saved_by"] = self.device_id
            df["pred_ts"] = float(entry['ts'])
            df["decision_ts"] = decision_ts
            df["n_nodes"] = len(buffer)
            df["round_probs"] = round_probs
            frames.append(df)
        if not frames:
            return
        full_df = pd.concat(frames, ignore_index=True)
        try:
            if not os.path.exists(self.test_data_path):
                os.makedirs(self.test_data_path)
            write_deltalake(self.test_data_path, full_df, mode="append", schema_mode="merge")
            print(f"✅ [SUCESSO] {len(full_df)} linhas gravadas em {self.test_data_path}")
        except Exception as e:
            print(f"❌ [IO] Erro ao gravar em {self.test_data_path}: {e}")

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

        device = data.get("device", str(node_id))
        self.m_msg_delay.labels(node=device, mode=self.mode).observe(max(0.0, time.time() - msg_ts))

        entry = {'ts': msg_ts, 'data': test_data, 'pred_probs': pred_probs, 'preds': preds, 'device': device}
        if node_id not in buffer or msg_ts > buffer[node_id]["ts"]:
            buffer[node_id] = entry

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

            if len(current_round_buffer) == 0:
                print("⚠️ [AGG] Janela fechou sem dados válidos.")
                continue

            # FL: só o servidor central agrega. Os workers também recebem a sua própria
            # predição (o broker local entrega-a), mas não decidem nada.
            is_central = self.peer_ip == self.server_ip
            if self.mode == "federated" and not is_central:
                print("👷 [AGG] FL: sou worker, a decisão é do servidor central.")
                continue

            self.pred_probs_dict = {k: v['pred_probs'] for k, v in current_round_buffer.items()}
            # Um erro numa ronda não pode matar o worker (thread) — regista e segue
            try:
                final_probs = self._run_aggregation(self.pred_probs_dict)
            except Exception as e:
                print(f"❌ [AGG] Erro na agregação, ronda ignorada: {e}")
                self.pred_probs_dict = {}
                continue
            decision_ts = time.time()

            # final_probs pode ser lista (avg) ou np.ndarray -> não usar "if final_probs"
            final_prob_idx = (
                int(np.argmax(final_probs[0]))
                if final_probs is not None and len(final_probs) > 0
                else None
            )
            final_label = self.reverse_label_map.get(final_prob_idx, "UNKNOWN")
            print(f"🌐 [AGG] Decisão global ({self.mode}, {self.agg_method}): {final_label}")

            # --- Métricas Prometheus ---
            m_labels = dict(node=self.device_id, mode=self.mode, method=self.agg_method)
            self.m_rounds.labels(**m_labels).inc()
            self.m_decisions.labels(**m_labels, decision=final_label).inc()
            self.m_nodes.labels(node=self.device_id).set(len(current_round_buffer))
            if self.node_id in current_round_buffer:
                own = current_round_buffer[self.node_id]
                if self.reverse_label_map.get(own['preds'][0]) != final_label:
                    self.m_divergence.labels(**m_labels).inc()
            for entry in current_round_buffer.values():
                self.m_decision_delay.labels(node=entry['device'], mode=self.mode).observe(
                    max(0.0, decision_ts - entry['ts'])
                )

            # --- Gravação para a análise offline ---
            # GL: cada nó grava a SUA linha. FL: o central grava as de todos os nós.
            if self.phase in ["INFERENCE", "METRICS"]:
                if self.mode == "federated":
                    nodes_to_save = list(current_round_buffer.keys())
                else:
                    nodes_to_save = [self.node_id] if self.node_id in current_round_buffer else []
                self._save_round(current_round_buffer, nodes_to_save, final_label, final_probs, decision_ts)

            # --- Envio da decisão global ---
            payload = {
                "id": self.broker_id,
                "ts": decision_ts,
                "final_pred": final_label,
            }
            if self.mode == "federated":
                targets = [p[0] for p in self.current_peer_list]  # central -> todos os workers
            else:
                targets = resolve_targets_by_index(self.current_peer_list, self.aggregation_dest_indices)
            for ip in targets:
                self.mqtt_com.publish(payload, topic=f"{ip.replace('.', '_')}/train")
            self.pred_probs_dict = {}

if __name__ == "__main__":
    aggregator = Aggregator()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass