from aggregation_algs.algs import ALGS_DICT
from yaml import Loader, load
from client.mqtt_layer import Communication_Layer
import json, threading, time, uuid
import warnings
warnings.filterwarnings("ignore")

from aggregation_algs.aggregation_utils import resolve_targets_by_index

class Aggregator:
 
    def __init__(self):

        with open("client/config.yaml", "r") as file:
            self.config = load(file, Loader=Loader)
 
        self.mosquitto_port = self.config["mosquitto_port"]
        self.broadcast_port = self.config["broadcast_port"]
        self.broadcast_mask = self.config["broadcast_mask"]
        self.peer_ip = self.config["peer_ip"]
        self.broker_id = self.peer_ip.replace(".", "_")

        self.mode = self.config["mode"]
        self.server_ip = self.config["central_server"]
        self.server_id = self.server_ip.replace(".", "_")

        self.current_peer_list = []
        self.aggregation_dest_indices = self.config["routing_topology"]["aggregation_topology"]

        if self.mode == "federated":
            if self.peer_ip == self.server_ip:
                print("[AGGREGATOR] Eu sou o SERVIDOR CENTRAL (Main).")
                self._setup_mqtt_client(subscribe_topic="+/agg")
                self._start_agg_worker()
            else:
                print("[AGGREGATOR] Modo Federated: Sou um Worker.")
                pass 
        else: 
            self._setup_mqtt_client(subscribe_topic="+/agg")
            self._start_agg_worker()

        # self._setup_mqtt_client()
        # self._start_agg_worker()

        self.remote_params = {}

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
        self.remote_params = self.mqtt_com.subscribe(topic=subscribe_topic)
        self.mqtt_com.subscribe("system/peers")

    def _start_agg_worker(self):
        agg_thread = threading.Thread(target=self.agg_worker)
        agg_thread.start()

    def aggregate(self, params_dict, method):
        """
        Agrega os hiperparâmetros recebidos de diferentes nós usando o método especificado.
        Args:
            params_dict: Dicionário contendo hiperparâmetros de diferentes nós
            method: Método de agregação a ser usado ('avg' ou 'majority')
        Returns:
            aggregated_params: Dicionário com os hiperparâmetros agregados
        """
        if method not in ALGS_DICT:
            raise ValueError(f"Método de agregação '{method}' não suportado.")

        aggregate_function = ALGS_DICT[method]
        aggregated_params = aggregate_function(params_dict)

        return aggregated_params

    def agg_worker(self):
        """
        Worker que processa mensagens recebidas e realiza a agregação.
        """
        while True:
            topic, data = self.mqtt_com.msg_queue.get()

            if topic == "system/peers":
                print(f"[{self.broker_id}] RECEIVED on {topic}: {data}")
                self.current_peer_list = data
                print(f"[AGGREGATION] Lista de peers atualizada: {self.current_peer_list}")
                self.mqtt_com.msg_queue.task_done()
                continue
            else:    
                print(f"[{self.broker_id}] RECEIVED on {topic}: {data}")
                if "trained_params" not in data:
                    self.mqtt_com.msg_queue.task_done()
                    continue

                node_id = data["id"]
                params = data["trained_params"]
                self.remote_params[node_id] = params
                aggregated_params = self.aggregate(self.remote_params, method="avg")
                payload = {
                    "id": self.broker_id,
                    "agg_params": aggregated_params
                }
                print("Aggregated Params", payload)
                targets = resolve_targets_by_index(self.current_peer_list, self.aggregation_dest_indices)
                if not targets:
                    self.mqtt_com.publish(payload, topic=f"{self.broker_id}/train")
                else:
                    for ip in targets:
                        target_id = ip.replace(".", "_")
                        self.mqtt_com.publish(payload, topic=f"{target_id}/train")
                self.mqtt_com.msg_queue.task_done()

if __name__ == "__main__":
    aggregator = Aggregator()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass