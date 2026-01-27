from data_utils import build_param_grid, data_split, load_data, resolve_targets_by_index
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yaml import Loader, load
import json, threading, time, uuid

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

        self.df = load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = data_split(self.df)

        self._setup_mqtt_client()
        self._start_pipe_worker()

        with open("param_config.yaml", "r") as file:
            self.config_param = load(file, Loader=Loader)

        self.param_grid = build_param_grid(self.config_param["param_grid"])
        # threading.Thread(target=self.run_pipeline).start()
        self.best_model = None
        self.best_params = None

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

    def build_pipeline(self, scaler, model):
        """
        Editar consoante o pré-processamento necessário
        Constroi a pipeline de classificação
        Returns:
            pipeline: A sklearn Pipeline object
        """
        pipeline = Pipeline([("scaler", scaler), ("classifier", model)])
        return pipeline

    def param_tuning(self, pipeline, param_grid, X_train, y_train):
        """
        Executa o GridSearchCV numa pipeline
        Treina  e faz hyperparameter tuning no modelo tedno em conta a param_grid
        Args:
            pipeline: A sklearn Pipeline object
            param_grid: Dicionário com os hyperparâmetros (param_config.yaml)
            X_train: Features de treino
            y_train: Labels de treino
            X_test: Features de teste
            y_test: Labels de teste
        Returns:
            best_params: Melhores parâmetros encontrados
            grid_search: O melhor modelo encontrado, já treinado
        """
        grid_search_model = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        print("GRID", self.param_grid)
        grid_search_model.fit(X_train, y_train)
        best_params = grid_search_model.best_params_

        return best_params, grid_search_model

    def evaluate(self, model, X_train, X_test, y_train, y_test):
        """
        Recebe o modelo treinado do param_tuning
        Avalia o modelo nos dados de treino e teste
        Nota: o score() -> predict + accuracy_score
        Returns:
            train_accuracy: Acurácia nos dados de treino
            test_accuracy: Acurácia nos dados de teste
        """
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        return train_accuracy, test_accuracy

    def run_pipeline(self):
        '''
        Executa a pipeline de treino com os melhores parâmetros encontrados.
        1. Constroi a pipeline.
        2. Realiza o hyperparameter tuning usando a grid adaptativa.
        3. Treina o modelo com os melhores parâmetros.
        4. Publica os parâmetros treinados para os destinos definidos.
        5. Avalia o modelo treinado nos dados de treino e teste.
        6. Atualiza o estado de treino.
        '''
        self.current_round += 1
        pipeline = self.build_pipeline(
            scaler=StandardScaler(), model=RandomForestClassifier()
        )
        self.best_params, self.best_model = self.param_tuning(
            pipeline, self.param_grid, self.X_train, self.y_train
        )
        trained_params_payload = {
            "id": self.peer_ip,
            "trained_params": self.best_params,
            "ts": time.time()
        }
        if self.mode == "federated":
            self.mqtt_com.publish(trained_params_payload, topic=f"{self.broker_id}/agg")
        else:
            targets = resolve_targets_by_index(self.current_peer_list, self.pipeline_dest_indices)
            if not targets:
                self.mqtt_com.publish(trained_params_payload, topic=f"{self.broker_id}/agg")
            else:
                for ip in targets:
                    target_id = ip.replace(".", "_")
                    self.mqtt_com.publish(trained_params_payload, topic=f"{target_id}/agg")
        train_acc, test_acc = self.evaluate(
            self.best_model, self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.is_training = False
        print("Terminei o treino!")

    def create_adaptive_grid(self, origin_params, spread=0.2):
        """
        Gera uma nova grid de pesquisa centrada nos valores recebidos.
        spread=0.2 significa que vamos procurar 20% acima e 20% abaixo.
        Args:
            origin_params: Dicionário com os parâmetros recebidos
            spread: Percentagem de variação para criar a grid
        Returns:
            new_grid: Dicionário com a nova grid de pesquisa
        """
        new_grid = {}
        
        for param, value in origin_params.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, int):
                    lower = int(value * (1 - spread))
                    upper = int(value * (1 + spread))
                    
                    # impede que arredonde para 0 ou negativo
                    lower = max(1, lower)
                    upper = max(1, upper)

                    values = sorted(list(set([max(1, lower), value, upper])))
                    new_grid[param] = values
                else:
                    lower = value * (1 - spread)
                    upper = value * (1 + spread)
                    # impede que arredonde para 0 ou negativo
                    lower = max(1, lower)
                    upper = max(1, upper)

                    new_grid[param] = [int(lower), int(value), int(upper)]
            else:
                new_grid[param] = [int(value)]
        return new_grid

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
                    new_params = data['agg_params']
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

                    self.param_grid = self.create_adaptive_grid(new_params)
                    self.is_training = True
                    self.run_pipeline()
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