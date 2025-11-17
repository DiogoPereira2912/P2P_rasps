import json, queue
from paho.mqtt import client as mqtt_client

class Communication_Layer:
    """
        Camada de comunicação MQTT full mesh para publicar e subscrever mensagens.

        Cada broker publica mensagens no sub-tópico `<client_id>/...`
        e subscreve `#` para receber mensagens de outros brokers, ignorando
        mensagens enviadas pelo próprio client_id.

        Attributes:
            client_id (str): Identificador único do cliente MQTT.
            base_topic (str): Prefixo base para tópicos MQTT.
            qos (int): Qualidade do serviço MQTT (0, 1 ou 2).
            client (paho.mqtt.client.Client): Instância do cliente MQTT.
    """
    def __init__(
        self,
        broker="",
        port=1884,
        client_id="",
        user="admin",
        pwd="public",
        base_topic="",
        qos=0
    ):
        """
        broker: host do broker
        port: porta do broker
        client_id: identificador MQTT
        base_topic: prefixo base para publicação
        qos: qualidade do serviço MQTT
        """
        self.client_id = client_id
        self.base_topic = base_topic
        self.qos = qos
        self.client = self._connect_mqtt(broker, port, user, pwd)
        self.client.loop_start()

    def _connect_mqtt(self, broker, port, user, pwd):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"[{self.client_id}] Connected to {broker}:{port}")
            else:
                print(f"[{self.client_id}] Connection failed with code {rc}")

        def on_disconnect(client, userdata, rc):
            print(f"[{self.client_id}] Disconnected, rc={rc}")

        client = mqtt_client.Client(client_id=self.client_id)
        if user and pwd:
            client.username_pw_set(user, pwd)
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.connect(broker, port)
        return client

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, payload, topic):
        full_topic = f"{self.base_topic}{topic}"
        result = self.client.publish(full_topic, json.dumps(payload), qos=self.qos)
        if result[0] != 0:
            print(f"[{self.client_id}] Failed to publish to {full_topic}")

    def subscribe(self, topic):
        def on_message(client, userdata, msg):
            if msg.topic.startswith(f"{self.client_id}/"):
                return
            data = json.loads(msg.payload.decode())
            print(f"[{self.client_id}] RECEIVED on {msg.topic}: {data}")
            return data
        self.client.subscribe(topic, qos=self.qos)
        self.client.on_message = on_message