import threading, time, uuid, os, json, sys
import pandas as pd
import numpy as np
from yaml import Loader, load
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from client.mqtt_layer import Communication_Layer
from labeling_utils import process_df, merge_labelled_dfs, process_labelled_df, load_class_mappings
from metrics import MetricsAnalyzer

RULES_LOCAL_PATH = "labeling/rules.yaml"
RULES_GLOBAL_PATH = "labeling/global_rules.yaml"

class Labeller:

    def __init__(self, device_id):
        self.device_id = device_id
        
        # --- Configs ---
        with open(RULES_LOCAL_PATH, "r") as file:
            self.rules = load(file, Loader=Loader)

        with open(RULES_GLOBAL_PATH, "r") as file:
            self.global_rules = load(file, Loader=Loader)

        with open("client/config.yaml", "r") as file:
            self.config = load(file, Loader=Loader)

        # --- Rede ---
        self.mosquitto_port = self.config["mosquitto_port"]
        self.peer_ip = self.config["peer_ip"]
        self.broker_id = self.peer_ip.replace(".", "_")
        self.current_peer_list = []
        self.min_peers = self.config['min_peers']

        # --- Caminhos ---
        self.parquet_raw_path = f"data_exports/raw_{self.config['device_id']}"
        # self.parquet_labelled_path = f"data_exports/labelled_{self.config['device_id']}"
        self.output_folder = f"data_exports/global_output_{self.config['device_id']}"
        # self.output_folder_local = f"data_exports/local_output_{self.config['device_id']}"
        self.inf_output_folder = f"data_exports/inference_output_{self.config['device_id']}"
        self.inf_local_data_path = f"data_exports/local_inf_data_{self.config['device_id']}"
        self.results_path = f"results/"
        
        # --- ESTRATÉGIA BATCH FINAL ---
        self.local_full_history = []  
        self.peer_history = {}        
        self.batch_count = 0          
        self.TARGET_BATCHES = 10    
        self.is_finished = False      
        self.merge_done = False    
        self.phase = "IDLE"   

        # --- GROUNDTRUTH INFERENCE ---
        self.local_groundtruth_history = []
        self.inf_batch_count = 0          
        self.inf_TARGET_BATCHES = 10   

        # --- MQTT ---
        self._setup_mqtt_client()
        self._start_label_worker()

    def _setup_mqtt_client(self):
        self.mqtt_com = Communication_Layer(
            broker=self.peer_ip,
            port=self.mosquitto_port,
            client_id=f"labeller_{self.broker_id}_{str(uuid.uuid4())[:4]}",
            base_topic="",
            qos=1,
        )
        self.mqtt_com.subscribe("system/control/#") 
        self.mqtt_com.subscribe("system/peers")
        self.mqtt_com.subscribe("internal/raw_detections")
        self.mqtt_com.subscribe("+/dataset")
        self.mqtt_com.subscribe("system/inference")
        self.mqtt_com.client.message_callback_add("system/control/#", self.on_control_message)

    def on_control_message(self, client, userdata, msg):
        """
        Muda o comportamento do Labeller consoante a ordem do Streamlit
        """
        try:
            payload = json.loads(msg.payload.decode())
            cmd = payload.get("command")
            
            self.phase = cmd

            if self.phase == "COLLECTION":
                self.local_full_history = []
                self.peer_history = {}
                self.batch_count = 0
                self.is_finished = False
                self.merge_done = False
                
                cfg = payload.get("config", {})
                self.TARGET_BATCHES = cfg.get("max_saves", 10)

            elif self.phase == "INFERENCE":

                self.local_groundtruth_history = []
                self.inf_batch_count = 0
                self.is_finished = False
                self.merge_done = False 
                self.peer_history = {}

                cfg = payload.get("config", {})
                self.inf_TARGET_BATCHES = cfg.get("max_saves", 10)
            
            elif self.phase == "METRICS":
                print("📊 FASE ALTERADA PARA: METRICS no Labeler!")
                self.mqtt_com.msg_queue.put(("generate_metrics", {}))

        except Exception as e:
            print(f"Erro no controlo Labeler: {e}")

    def check_rule(self, row, condicoes):
        if 'confidence_min' in condicoes:
            conf = row.get(f'confidence', 0)
            if conf < condicoes['confidence_min']: return 0
        if 'ROI_rule' in condicoes:
            x1, x2 = row.get('box_x1', 0), row.get('box_x2', 0)
            y1, y2 = row.get('box_y1', 0), row.get('box_y2', 0)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx1, rx2 = condicoes['ROI_rule']['x1'], condicoes['ROI_rule']['x2']
            ry1, ry2 = condicoes['ROI_rule']['y1'], condicoes['ROI_rule']['y2']
            if not ((rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)): return 0
        if 'pose_rule' in condicoes:
            kpts = row.get('keypoints', [])
            if not isinstance(kpts, list) or len(kpts) < 17: return 0
            p_rule = condicoes['pose_rule']
            idx_a, idx_b = p_rule['ponto_a'], p_rule['ponto_b']
            max_dist = p_rule['max_dist']
            pa, pb = kpts[idx_a], kpts[idx_b]
            if len(pa) > 2 and (pa[2] < 0.001 or pb[2] < 0.001): return 0
            dist = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
            if dist > max_dist: return 0
        return 1

    def apply_binary_labels(self, row, rules, device_id):
        result = {}
        for regra in rules['regras']:
            nome = f"{regra['label']}_{device_id}"
            result[nome] = 1 if self.check_rule(row, regra['condicoes']) == 1 else 0      
        return result

    def get_global_status(self, row, rules_config, device_ids:list):
        current_status = "SAFE"
        min_priority = 999
        for rule in rules_config.get('regras', []): 
            label_target = rule['label']
            condicoes = rule['condicoes']
            prioridade = rule['prioridade']
            rule_match = True
            for id in condicoes:
                if id not in device_ids:
                    rule_match = False
                    continue
                else:
                    req_r1 = condicoes.get(id)
                    val_r1 = row.get(f"{label_target}_{id}", 0) 
                    match_r1 = (val_r1 == req_r1) if req_r1 is not None else True
                if not match_r1:
                    rule_match = False
                    break
            if rule_match:
                if prioridade < min_priority:
                    min_priority = prioridade
                    current_status = rule['label']
        return current_status

    def process_batch(self, data):

        search_ids = self.current_peer_list + [self.device_id]
        print(f"Processando batch com dados de: {search_ids}")

        df_raw = pd.DataFrame(data)
        if 'bbox_coords' in df_raw.columns:
            df_raw[['box_x1', 'box_y1', 'box_x2', 'box_y2']] = pd.DataFrame(
                df_raw['bbox_coords'].tolist(), 
                index=df_raw.index
            )
            df_raw = df_raw.drop(columns=['bbox_coords'])

        if self.phase == "COLLECTION":
            if self.is_finished: 
                return

            # 1. Gravar Raw locais
            try:
                if not os.path.exists(self.parquet_raw_path):
                    os.makedirs(self.parquet_raw_path)
                write_deltalake(self.parquet_raw_path, df_raw, mode="append")
            except Exception as e:
                print(f"Erro ao gravar Labelled local: {e}")

            # 2. Processar Labelled -> regras locais
            df_working = process_df(df_raw.copy())
            binary_cols = df_working.apply(
                lambda row: self.apply_binary_labels(row, self.rules, self.device_id), axis=1
            )
            binary_df = pd.DataFrame(binary_cols.tolist(), index=df_working.index)
            df_labelled = pd.concat([df_working[['ts']], binary_df], axis=1)

            # 3. Gravar Labelled locais
            # try:
            #     if not os.path.exists(self.parquet_labelled_path):
            #         os.makedirs(self.parquet_labelled_path)
            #     write_deltalake(self.parquet_labelled_path, df_labelled, mode="append")
            # except Exception as e:
            #     print(f"Erro ao gravar Labelled local: {e}")

            df_export = df_labelled.copy()
            df_export['ts'] = df_export['ts'].astype(str)
            batch_records = df_export.to_dict(orient='records')
            
            self.local_full_history.extend(batch_records)
            self.batch_count += 1
            
            print(f"[{self.device_id}] Batch {self.batch_count}/{self.TARGET_BATCHES} acumulado.")

            if self.batch_count >= self.TARGET_BATCHES:
                self.is_finished = True
                print(f"🏁 RECOLHA TERMINADA. A enviar pacote final para a rede...")
                
                payload = {
                    "id": self.device_id,
                    "ts": time.time(),
                    "labelled_df": self.local_full_history,
                }
                self.mqtt_com.publish(payload, f"{self.broker_id}/dataset")
                print("🚀 PACOTE ENVIADO. À espera dos dados para Merge Final...")

                df_final_local = pd.DataFrame(self.local_full_history)
                df_final_local['GLOBAL_STATUS'] = df_final_local.apply(
                    lambda row: self.get_global_status(row, self.global_rules, search_ids), axis=1
                )

                # if not os.path.exists(self.output_folder_local):
                #     os.makedirs(self.output_folder_local)
                # write_deltalake(self.output_folder_local, df_final_local, mode="append")

        elif self.phase == "INFERENCE":

            df_working = process_df(df_raw.copy())
            binary_cols = df_working.apply(
                lambda row: self.apply_binary_labels(row, self.rules, self.device_id), axis=1
            )
            binary_df = pd.DataFrame(binary_cols.tolist(), index=df_working.index)
            df_labelled = pd.concat([df_working[['ts']], binary_df], axis=1)
            df_labelled['ts'] = df_labelled['ts'].astype(str)
            df_labelled_inf = df_labelled.to_dict(orient='records')
            payload = {
                "id": self.device_id,
                "ts": time.time(),
                "labelled_df": df_labelled_inf,
            }
            self.mqtt_com.publish(payload, f"system/inference")
            print("🚀 PACOTE ENVIADO PARA INFERENCIA")
            
            self.local_groundtruth_history.extend(df_labelled_inf)
            self.inf_batch_count += 1

            if self.inf_batch_count >= self.inf_TARGET_BATCHES:
                self.is_finished = True

                print(f"🏁 RECOLHA TERMINADA. A enviar pacote final para a rede...")
                
                payload = {
                    "id": self.device_id,
                    "ts": time.time(),
                    "labelled_df": self.local_groundtruth_history,
                }
                self.mqtt_com.publish(payload, f"{self.broker_id}/dataset")
                print("🚀 PACOTE ENVIADO. À espera dos dados para Merge Final...")
    
    def label_worker_on_message(self):
        while True:
            if self.merge_done and self.phase == "COLLECTION":
                time.sleep(1)
                continue

            topic, data = self.mqtt_com.msg_queue.get()

            if topic == "system/peers":
                self.current_peer_list = [peer[2] for peer in data]
                print(f"📡 Lista de peers atualizada: {self.current_peer_list}")
                self.mqtt_com.msg_queue.task_done()
                continue

            if topic == "generate_metrics" and self.phase == "METRICS":
                time.sleep(0.5)
                print("🔍 TESTE DE MÉTRICAS COM DADOS DE INFERÊNCIA E GROUND TRUTH")
                try:
                    metrics_analyzer = MetricsAnalyzer(
                        data_path_local=self.inf_local_data_path,
                        data_path_global=self.inf_output_folder,
                        results_path=self.results_path,
                        device_id=self.device_id
                    )
                    metrics_analyzer.run_analysis()
                    print("✅ Gráficos gerados com sucesso!")
                except Exception as e:
                    print(f"❌ Erro nas métricas: {e}")
                self.mqtt_com.msg_queue.task_done()
                continue

            if self.phase == "COLLECTION" or self.phase == "INFERENCE":
                if topic == "internal/raw_detections":
                    self.process_batch(data)
                    self.mqtt_com.msg_queue.task_done()
                    continue
            
            if "dataset" in topic:
                if self.phase != "COLLECTION" and self.phase != "INFERENCE":
                    self.mqtt_com.msg_queue.task_done()
                    continue

                try:
                    search_ids = self.current_peer_list + [self.device_id]
                    node_id = data.get("id")
                    if node_id == self.device_id or not data.get("labelled_df"):
                        self.mqtt_com.msg_queue.task_done(); 
                        continue

                    remote_labelled_df = pd.DataFrame(data["labelled_df"])
                    if 'ts' in remote_labelled_df.columns:
                        remote_labelled_df['ts'] = pd.to_datetime(remote_labelled_df['ts'])

                    self.peer_history[node_id] = remote_labelled_df

                    if self.is_finished and self.phase == "COLLECTION":

                        my_df = pd.DataFrame(self.local_full_history)
                        if 'ts' in my_df.columns: # save outputs locally with global labels
                            my_df['ts'] = pd.to_datetime(my_df['ts'])
                        
                        dfs_to_merge = [process_labelled_df(my_df)]
                        for pid, pdf in self.peer_history.items():
                            if not pdf.empty:
                                dfs_to_merge.append(process_labelled_df(pdf))

                        if len(dfs_to_merge) >= (self.min_peers + 1):  # +1 para incluir o próprio dispositivo
                            print("A GERAR DATASET MERGED FINAL")
                            df_final = merge_labelled_dfs(dfs_to_merge)
                            df_final['GLOBAL_STATUS'] = df_final.apply(
                                lambda row: self.get_global_status(row, self.global_rules, search_ids), axis=1
                            )
                            df_final['GLOBAL_STATUS'] = df_final['GLOBAL_STATUS'].astype(str)
                            if not os.path.exists(self.output_folder):
                                os.makedirs(self.output_folder)
                            
                            write_deltalake(self.output_folder, df_final, mode="append")
                            self.merge_done = True                            
                            print(f"✅✅✅ MERGE GRAVADO COM SUCESSO! ({len(df_final)} linhas)")
                            print("🛑 TRABALHO CONCLUÍDO. A ENCERRAR O PROCESSO.")

                    elif self.is_finished and self.phase == "INFERENCE":
                                                              
                        my_df = pd.DataFrame(self.local_groundtruth_history)
                        if 'ts' in my_df.columns: # save outputs locally with global labels
                            my_df['ts'] = pd.to_datetime(my_df['ts'])
                        
                        dfs_to_merge = [process_labelled_df(my_df)]
                        for pid, pdf in self.peer_history.items():
                            if not pdf.empty:
                                dfs_to_merge.append(process_labelled_df(pdf))

                        if len(dfs_to_merge) >= (self.min_peers + 1):
                            print("A GERAR DATASET MERGED FINAL")
                            df_final = merge_labelled_dfs(dfs_to_merge)
                            df_final['GLOBAL_STATUS'] = df_final.apply(
                                lambda row: self.get_global_status(row, self.global_rules, search_ids), axis=1
                            )
                            df_final['GLOBAL_STATUS'] = df_final['GLOBAL_STATUS'].astype(str)
                            if not os.path.exists(self.inf_output_folder):
                                os.makedirs(self.inf_output_folder)
                            
                            write_deltalake(self.inf_output_folder, df_final, mode="append")
                            print(f"✅✅✅ MERGE INFERENCIA GRAVADO COM SUCESSO! ({len(df_final)} linhas)")
                            print("🛑 TRABALHO CONCLUÍDO. A ENCERRAR O PROCESSO.")

                            self.local_groundtruth_history = [] 
                            self.inf_batch_count = 0
                            self.is_finished = False
                            self.peer_history = {}

                except Exception as e:
                    print(f"❌ Erro no Worker: {e}")
        
            self.mqtt_com.msg_queue.task_done()

    def _start_label_worker(self):
        label_thread = threading.Thread(target=self.label_worker_on_message, daemon=True)
        label_thread.start()

if __name__ == "__main__": 
    
    with open("client/config.yaml", "r") as file:
        config = load(file, Loader=Loader)
    device_id = config["device_id"]
    service = Labeller(device_id)
    print(f"🏭 Labeller Service ({device_id}) iniciado e à espera de ordens...")
    while True:
        time.sleep(1)