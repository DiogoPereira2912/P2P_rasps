from data_utils import data_split, load_data, resolve_targets_by_index, load_class_mappings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yaml import Loader, load
import json, threading, time, uuid, pickle, os
from client.mqtt_layer import Communication_Layer
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
 
MODELS = {'RandomForest': RandomForestClassifier}

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

        self.is_training = False
        self.current_peer_list = []
        self.min_peers = self.config["min_peers"]
        self.pipeline_dest_indices = self.config["routing_topology"]["pipeline_topology"]

        self.label_map, _, self.class_list = load_class_mappings(yaml_path="labeling/global_rules.yaml")
        self.models_dump_path = f"models/"
        self.loaded_model = None
        self.phase = None

        self._setup_mqtt_client()
        self._start_pipe_worker()

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
        self.mqtt_com.subscribe("system/inference")
        self.mqtt_com.subscribe("system/control/#")
        self.mqtt_com.client.message_callback_add("system/control/#", self.on_control_phase_message)

    def on_control_phase_message(self, topic, userdata, msg):
        try:
            print(f"📩 MENSAGEM RECEBIDA: {msg.payload}")
            payload = json.loads(msg.payload.decode())
            cmd = payload.get("command")
            
            self.phase = cmd
            print(f"🔄 FASE ALTERADA PARA: {self.phase}")

            if cmd == "TRAIN":
                threading.Thread(target=self.run_model_training).start()
            
            elif cmd == "INFERENCE":
                if not self.loaded_model:
                    self._try_load_existing_model()
                    if not self.loaded_model:
                        print("⚠️ AVISO: Modo Inferência ativado mas SEM MODELO treinado.")

            elif cmd == "STOP":
                print("🛑 Paragem recebida.")
                
        except Exception as e:
            print(f"Erro ao processar comando: {e}")

    def load_model_pkl(self, filepath):
        """
        Carrega um modelo sklearn a partir de um ficheiro pickle
        Args:
            filepath: Caminho para o ficheiro pickle
        Returns:
            model: O modelo carregado
        """
        try:
            if os.path.exists(filepath):
                model = pickle.load(open(filepath, 'rb'))
                return model
        except Exception as e:
            print(f"Erro ao carregar o modelo do ficheiro {filepath}: {e}")
            return None
        
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
        self.mqtt_com.subscribe(topic="+/train")

    def build_pipeline(self, scaler, model):
        """
        Editar consoante o pré-processamento necessário
        Constroi a pipeline de classificação
        Returns:
            pipeline: A sklearn Pipeline object
        """
        pipeline = Pipeline([("scaler", scaler), ("classifier", model)])
        return pipeline

    def param_tuning(self, pipeline, X_train, y_train):
        """
        Executa o GridSearchCV numa pipeline
        Treina  e faz hyperparameter tuning no modelo tedno em conta a param_grid
        Args:
            pipeline: A sklearn Pipeline object
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
            param_grid = { 
                'classifier__n_estimators': [200, 500],
                'classifier__max_features': ['auto', 'sqrt', 'log2'],
                'classifier__max_depth' : [4,5,6,7,8],
                'classifier__criterion' :['gini', 'entropy']
            },
            cv=5,  # 5-fold cross-validation
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid_search_model.fit(X_train, y_train)
        best_params = grid_search_model.best_params_

        return best_params, grid_search_model

    def evaluate(self, model, X_train, X_test, y_train, y_test):
        """
        Avalia o modelo nos dados de treino e teste
        Nota: o score() -> predict + accuracy_score
        Returns:
            train_accuracy: Acurácia nos dados de treino
            test_accuracy: Acurácia nos dados de teste
        """
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        return train_accuracy, test_accuracy

    def run_model_training(self):
        '''
            Executa a pipeline de treino com os melhores parâmetros encontrados.
            1. Constroi a pipeline.
            2. Realiza o hyperparameter tuning usando a grid adaptativa.
            3. Treina o modelo com os melhores parâmetros.
            4. Avalia o modelo nos dados de treino e teste.
            5. Salva o modelo treinado em um ficheiro pickle.
        '''
        df = load_data()
        X_train, X_test, y_train, y_test = data_split(df, self.label_map)
        for name, model_class in MODELS.items():
            pipeline = self.build_pipeline(
                scaler=StandardScaler(), model= model_class()
            )
            self.best_params, self.best_model = self.param_tuning(
                pipeline, X_train, y_train
            )
            # filename = f"best_{name}_{self.broker_id}.pkl"
            # filepath = self.models_dump_path + filename
            # pickle.dump(self.best_model, open(filepath, 'wb'))
        
        self.loaded_model = self.best_model
        print("✅ Modelo treinado com sucesso.")

    def run_classification(self, data):
        '''
            Executa a classificação nos dados de teste e publica as predições.
            1. Força o cálculo das probabilidades e predições.
            2. Prepara a payload com as predições.
            3. Publica as predições para os destinos definidos.
        '''
        df = pd.DataFrame(data)
        df = df.drop(columns=['ts'])
        pred_probs = self.loaded_model.predict_proba(df) # forçar cálculo de probabilidades
        preds = self.loaded_model.predict(df) # forçar cálculo de predições
        model_preds_payload = {
            "id": self.node_id,
            "ts": time.time(),
            "data": data,
            "pred_probs": pred_probs.tolist(),
            "preds": preds.tolist()
        }
        print(f"📊 Publicando Predições: {pred_probs}")
        if self.mode == "federated":
            self.mqtt_com.publish(model_preds_payload, topic=f"{self.broker_id}/agg")
        else:
            targets = resolve_targets_by_index(self.current_peer_list, self.pipeline_dest_indices)
            if not targets:
                self.mqtt_com.publish(model_preds_payload, topic=f"{self.broker_id}/agg")
            else:
                for ip in targets:
                    target_id = ip.replace(".", "_")
                    self.mqtt_com.publish(model_preds_payload, topic=f"{target_id}/agg")

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
                self.mqtt_com.msg_queue.task_done()
                continue

            elif topic == "system/inference":
                if self.phase == "INFERENCE":
                    if len(self.current_peer_list) >= self.min_peers:

                        node_id = data["id"]
                        msg_ts = data["ts"]
                        new_labelled_df = data["labelled_df"]

                        if node_id in last_processed_ts and msg_ts <= last_processed_ts[node_id]:
                            self.mqtt_com.msg_queue.task_done()
                            continue

                        last_processed_ts[node_id] = msg_ts

                        if self.loaded_model:
                            self.run_classification(new_labelled_df)
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