import threading, time, uuid, os, json, sys
import pandas as pd
import numpy as np
from yaml import Loader, load
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from client.mqtt_layer import Communication_Layer
from labeling.labeling_utils import process_df, merge_labelled_dfs, process_labelled_df

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

        # --- Caminhos ---
        self.parquet_raw_path = f"data_exports/raw_{self.config['device_id']}"
        self.parquet_labelled_path = f"data_exports/labelled_{self.config['device_id']}"
        self.output_folder = f"data_exports/global_output_{self.config['device_id']}"
        
        # --- ESTRATÉGIA BATCH FINAL ---
        self.local_full_history = []  
        self.peer_history = {}        
        
        self.batch_count = 0          
        self.TARGET_BATCHES = 10    
        self.is_finished = False      
        self.merge_done = False       

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
        self.mqtt_com.subscribe("#") 

    # --- (Funções de Regras - Sem alterações) ---
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
        for rule in rules_config.get('regras_globais', []): 
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
                    current_status = rule['nome']
        return current_status

    def process_batch(self, df_raw):

        if self.is_finished: 
            return

        # 1. Gravar Raw locais
        try:
            if not os.path.exists(self.parquet_raw_path):
                os.makedirs(self.parquet_raw_path)
            write_deltalake(self.parquet_raw_path, df_raw, mode="append")
        except Exception as e:
            print(f"Erro ao gravar Labelled local: {e}")

        # 2. Processar Labelled
        df_working = process_df(df_raw.copy())
        binary_cols = df_working.apply(
            lambda row: self.apply_binary_labels(row, self.rules, self.device_id), axis=1
        )
        binary_df = pd.DataFrame(binary_cols.tolist(), index=df_working.index)
        df_labelled = pd.concat([df_working[['ts']], binary_df], axis=1)

        # 3. Gravar Labelled locais
        try:
            if not os.path.exists(self.parquet_labelled_path):
                os.makedirs(self.parquet_labelled_path)
            write_deltalake(self.parquet_labelled_path, df_labelled, mode="append")
        except Exception as e:
            print(f"Erro ao gravar Labelled local: {e}")

        # 4. Acumular labelleds
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
            
            topic = f"{self.broker_id}/dataset"
            self.mqtt_com.publish(payload, topic)
            print("🚀 PACOTE ENVIADO. À espera dos dados dos vizinhos para Merge Final...")

    def label_worker_on_message(self):
        while True:
            if self.merge_done:
                time.sleep(1)
                continue

            topic, data = self.mqtt_com.msg_queue.get()

            if topic == "system/peers":
                self.current_peer_list = [peer[2] for peer in data]
                self.mqtt_com.msg_queue.task_done()
                continue
            
            if "dataset" in topic:
                try:
                    node_id = data.get("id")
                    if node_id == self.device_id or not data.get("labelled_df"):
                        self.mqtt_com.msg_queue.task_done(); continue

                    print(f"[{self.broker_id}] Recebido de {node_id} ({len(data['labelled_df'])} linhas)")
                    
                    remote_labelled_df = pd.DataFrame(data["labelled_df"])
                    if 'ts' in remote_labelled_df.columns:
                        remote_labelled_df['ts'] = pd.to_datetime(remote_labelled_df['ts'])

                    self.peer_history[node_id] = remote_labelled_df

                    if self.is_finished:
                        
                        my_df = pd.DataFrame(self.local_full_history)
                        if 'ts' in my_df.columns: my_df['ts'] = pd.to_datetime(my_df['ts'])
                        
                        dfs_to_merge = [process_labelled_df(my_df)]
                        for pid, pdf in self.peer_history.items():
                            if not pdf.empty:
                                dfs_to_merge.append(process_labelled_df(pdf))

                        if len(dfs_to_merge) >= 3:
                            
                            print("A GERAR DATASET MERGED FINAL")
                            df_final = merge_labelled_dfs(dfs_to_merge)
                            
                            df_final['GLOBAL_STATUS'] = df_final.apply(
                                lambda row: self.get_global_status(row, self.global_rules, self.current_peer_list), axis=1
                            )
                            df_final['GLOBAL_STATUS'] = df_final['GLOBAL_STATUS'].astype(str)

                            if not os.path.exists(self.output_folder):
                                os.makedirs(self.output_folder)
                            
                            write_deltalake(self.output_folder, df_final, mode="append")
                            
                            print(f"✅✅✅ MERGE GRAVADO COM SUCESSO! ({len(df_final)} linhas)")
                            print("🛑 TRABALHO CONCLUÍDO. A ENCERRAR O PROCESSO.")
                            
                            self.merge_done = True                            
                            sys.exit(0) 

                except Exception as e:
                    print(f"❌ Erro no Worker: {e}")
        
            self.mqtt_com.msg_queue.task_done()

    def _start_label_worker(self):
        label_thread = threading.Thread(target=self.label_worker_on_message, daemon=True)
        label_thread.start()

if __name__ == "__main__": 
    pass