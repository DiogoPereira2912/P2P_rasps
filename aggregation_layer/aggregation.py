from aggregation_algs.algs import ALGS_DICT
from yaml import Loader, load
from client.mqtt_layer import Communication_Layer
import json

# TODO: adcionar subscribe ao agg/

class Aggregator:
 
    def __init__(self):

        with open("client/config.yaml", "r") as file:
            self.config = load(file, Loader=Loader)

        # TODO adicionar base_topic para depois meter nas bridges dinamicas

        self.mosquitto_port = self.config["mosquitto_port"]
        self.broadcast_port = self.config["broadcast_port"]
        self.broadcast_mask = self.config["broadcast_mask"]
        self.peer_ip = self.config["peer_ip"]
        self.broker_id = self.peer_ip.replace(".", "_")

        self._setup_mqtt_client()

        self.remote_params = {
            "Node1": {
                "classifier__max_depth": 40,
                "classifier__min_samples_leaf": 1,
                "classifier__n_estimators": 200,
            },
            "Node2": {
                "classifier__max_depth": 40,
                "classifier__min_samples_leaf": 2,
                "classifier__n_estimators": 400,
            },
            "Node3": {
                "classifier__max_depth": 60,
                "classifier__min_samples_leaf": 3,
                "classifier__n_estimators": 600,
            },
        }

    def _setup_mqtt_client(self):
        """
        Cria o cliente MQTT e faz o subscribe ao tópico
        """
        self.mqtt_com = Communication_Layer(
            broker=self.peer_ip,
            port=self.mosquitto_port,
            client_id=f"pipeline_{self.broker_id}",
            base_topic="",
            qos=1,
        )
        # TODO: add subscribe do update via MQTT to train/
        self.remote_params = self.mqtt_com.subscribe(topic="agg/#")

    def aggregate(self, params_list, method):
        """
        Agrega os hiperparâmetros recebidos de diferentes nós usando o método especificado.
        Args:
            params_list: Lista de dicionários contendo hiperparâmetros de diferentes nós
            method: Método de agregação a ser usado ('avg' ou 'majority')
        Returns:
            aggregated_params: Dicionário com os hiperparâmetros agregados
        """
        if method not in ALGS_DICT:
            raise ValueError(f"Método de agregação '{method}' não suportado.")

        aggregate_function = ALGS_DICT[method]
        aggregated_params = aggregate_function(params_list)

        return aggregated_params


if __name__ == "__main__":

    aggregator = Aggregator()
    best_params = aggregator.remote_params
    aggregated_params_avg = aggregator.aggregate(best_params, method="avg")
    print("Aggregated Parameters (Average):", aggregated_params_avg)
    agg_params_payload = {"id": aggregator.peer_ip, "agg_params": aggregated_params_avg}
    aggregator.mqtt_com.publish(
        payload=agg_params_payload, topic=f"train/{aggregator.broker_id}"
    )

    aggregated_params_majority = aggregator.aggregate(best_params, method="majority")
    print("Aggregated Parameters (Majority):", aggregated_params_majority)
