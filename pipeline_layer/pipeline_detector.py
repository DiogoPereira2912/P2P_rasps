from data_utils import resolve_targets_by_index
from yaml import Loader, load
import json, threading, time, uuid, base64, io, torch
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore")

from client.mqtt_layer import Communication_Layer

 
class Model_Manager:

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

        self.current_round = 0
        self.is_training = False
        self.current_peer_list = []
        self.min_peers = self.config["min_peers"]
        self.pipeline_dest_indices = self.config["routing_topology"]["pipeline_topology"]

        # path to get model
        self.base_model_path = "../models/base/yolov8n.pt" 
        self.update_model_path = "../models/base/updated_model.pt"
        self.current_best_map = 0.0

        self._setup_mqtt_client()
        self._start_pipe_worker()

    def _setup_mqtt_client(self):
        """
        Cria o cliente MQTT e faz o subscribe ao tópico
        """
        self.mqtt_com = Communication_Layer(
            broker=self.peer_ip,
            port=self.mosquitto_port,
            client_id=f"pipeline_{self.broker_id}_{str(uuid.uuid4())[:4]}",
            base_topic="",
            qos=1,
        )
        if self.mode != "federated":
            self.mqtt_com.subscribe(topic="+/train")
        self.mqtt_com.subscribe("system/peers")

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

    def evaluate_local_baseline(self):
            """Valida o modelo atual para saber o meu ponto de partida."""
            print("📏 [INIT] A calcular mAP do modelo local...")
            try:
                local_model = YOLO(self.base_model_path)
                local_weights = local_model.model.state_dict()
                score, _, _ = self.validate_incoming_model(local_weights)
                
                self.current_best_map = score
                print(f"🏆 [INIT] O meu mAP inicial é: {self.current_best_map:.4f}")
                return self._serialize_model(local_weights)
            except Exception as e:
                print(f"❌ Erro ao avaliar baseline: {e}")
                return False

    def publish_weights(self, weights):
        """
        Extrai os pesos do modelo inicial 
        Converte os pesos para Base64 e envia via MQTT.
        """
        payload = {
            "id": self.broker_id,
            "ts": time.time(),
            "weights": weights 
        }
        if self.mode == "federated":
            self.mqtt_com.publish(payload, topic=f"{self.broker_id}/agg")
        else:
            targets = resolve_targets_by_index(self.current_peer_list, self.pipeline_dest_indices)
            if not targets:
                self.mqtt_com.publish(payload, topic=f"{self.broker_id}/agg")
            else:
                for ip in targets:
                    target_id = ip.replace(".", "_")
                    self.mqtt_com.publish(payload, topic=f"{target_id}/agg")
        print(f"📤 Modelo enviado para a rede")

    def validate_incoming_model(self, received_state_dict):
        '''
        Usa uma instancia vazia do modelo para carregar os pesos recebidos
        e valida o modelo com um dataset de validação.
        Retorna o mAP e os pesos validados.
        '''
        try:
            temp_model = YOLO(self.yolo_config_path, task='detect')
            temp_model.model.load_state_dict(received_state_dict, strict=False)
            
            metrics = temp_model.val(
                data="coco8.yaml", 
                imgsz=320, 
                batch=1, 
                workers=0, 
                device='cpu', 
                verbose=False, 
                plots=True
            )
            return metrics.box.map50, received_state_dict, temp_model
        except Exception as e:
            print(f"❌ Erro na validação: {e}")
            return 0.0, None, None

    def _verify_central_server(self, peers_list):
        """
        Verifica se sou o central
        Verifica se o servidor central está na lista de peers conhecidos.
        Se não estiver, adiciona uma bridge para ele.
        """
        if self.node_id == self.central_id:
            self.server_ip = self.peer_ip
            self.server_id = self.server_ip.replace(".", "_")
            print("MAIN SERVER")
        else:
            print("WORKER")
            for p in peers_list:
                if p[1] == self.central_id:
                    self.server_ip = p[0]
                    self.server_id = self.server_ip.replace(".", "_")
                    #print("SERVER ID", self.server_id)
        self.mqtt_com.subscribe(topic="+/train")

    def pipe_worker_on_message(self):
        """
        Worker para processar mensagens de pipeline recebidas via MQTT.
        1. Espera por mensagens no tópico de treino.
        2. Se receber parâmetros agregados, inicia o treino com esses parâmetros.
        3. Gera uma nova grid adaptativa baseada nos parâmetros recebidos.
        4. Executa a pipeline de treino.
        5. Publica os parâmetros treinados para os destinos definidos.
        """
        last_processed_ts = {}
        ## Dar inicio ao ciclo com o modelo base
        initial_weights_bytes = self.evaluate_local_baseline()
        if initial_weights_bytes:
            print("🚀 [BOOT] A enviar pesos iniciais para iniciar o ciclo...")
            initial_weights_b64 = base64.b64encode(initial_weights_bytes).decode('utf-8')
            self.publish_weights(initial_weights_b64)

        while True:
            topic, data = self.mqtt_com.msg_queue.get()

            if topic == "system/peers":
                self.current_peer_list = data
                print(f"[PIPELINE] Lista de peers atualizada: {self.current_peer_list}")
                print("PEERS CONHECIDOS:", len(self.current_peer_list))
                print("PEERS NECESSÁRIOS:", self.min_peers)
                if self.mode == "federated":
                    self._verify_central_server(self.current_peer_list)

                if len(self.current_peer_list) >= self.min_peers and not self.is_training:
                    # garante que só arranca o 1o treino quando tiver peers suficientes
                    self.is_training = True 
                    threading.Thread(target=self.run_pipeline).start() # arranca o 1o treino
                self.mqtt_com.msg_queue.task_done()
                continue

            else:
                if len(self.current_peer_list) >= self.min_peers:
                    node_id = data["id"]
                    msg_ts = data["ts"]

                    if node_id in last_processed_ts and msg_ts <= last_processed_ts[node_id]:
                        self.mqtt_com.msg_queue.task_done()
                        continue
                    last_processed_ts[node_id] = msg_ts

                    if self.is_training:
                        print(f"[{self.broker_id}] Já está em treino.")
                        self.mqtt_com.msg_queue.task_done()
                        continue
                    print(f"[{self.broker_id}] RECEIVED on {topic}: {data}")

                    self.is_training = True 
                    weights = data["agg_weights"]
                    if weights:
                        received_state_dict = self._deserialize_model(weights)
                        new_score, new_weights, new_model = self.validate_incoming_model(received_state_dict)
                        print(f"✅ Recebido de {node_id} | mAP: {new_score:.4f}")

                        if new_score >= self.current_best_map: # passar a agregação
                            print(f"🚀 Novo melhor modelo! mAP: {new_score:.4f} (antigo: {self.current_best_map:.4f})")
                            self.current_best_map = new_score
                            weights_to_send = self._serialize_model(new_weights)
                            self.publish_weights(weights_to_send) # publish new_weights to aggregation
                        else:
                            print(f"❌ Modelo rejeitado. mAP inferior ao atual ({self.current_best_map:.4f})")
                    self.mqtt_com.msg_queue.task_done()

    def _start_pipe_worker(self):
        '''
        Inicia o worker que processa mensagens de pipeline.
        '''
        pipe_thread = threading.Thread(target=self.pipe_worker_on_message)
        pipe_thread.start()

if __name__ == "__main__": 
    manager = Model_Manager()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass