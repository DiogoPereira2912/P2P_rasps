from data_utils import build_param_grid, data_split, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yaml import Loader, load
import json

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

        self._setup_mqtt_client()

        with open("param_config.yaml", "r") as file:
            self.config_param = load(file, Loader=Loader)

        self.param_grid = build_param_grid(self.config_param["param_grid"])
        self.best_model = None
        self.best_params = None

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
        self.mqtt_com.subscribe(topic="train/#")

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
        """
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        return train_accuracy, test_accuracy

    def run_pipeline(self):
        df = load_data()
        X_train, X_test, y_train, y_test = data_split(df)
        pipeline = self.build_pipeline(
            scaler=StandardScaler(), model=RandomForestClassifier()
        )
        self.best_params, self.best_model = self.param_tuning(
            pipeline, self.param_grid, X_train, y_train
        )
        trained_params_payload = {
            "id": self.peer_ip,
            "trained_params": self.best_params,
        }
        # TODO: publish best_params via MQTT to agg/
        self.mqtt_com.publish(
            payload=trained_params_payload, topic=f"{self.broker_id}/agg/"
        )
        train_acc, test_acc = self.evaluate(
            self.best_model, X_train, X_test, y_train, y_test
        )
        print("Best Parameters:", self.best_params)
        print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__": 
    manager = Model_Manager()
    manager.run_pipeline()