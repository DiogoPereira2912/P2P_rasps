import json, queue, uuid
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
        self.topic_broker_id = broker.replace(".", "_")
        self.base_topic = base_topic
        self.qos = qos
        
        self.msg_queue = queue.Queue()
        self.subscribed_topics = []

        self.client = self._connect_mqtt(broker, port, user, pwd)
        self.client.loop_start()
        self.client.on_message = self.on_message

    def _connect_mqtt(self, broker, port, user, pwd):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"[{self.client_id}] Ligado ao Broker!")
                for topic in self.subscribed_topics:
                    client.subscribe(topic, qos=self.qos)
                    print(f"[{self.client_id}] Re-subscrevi: {topic}")
            else:
                print(f"[{self.client_id}] Falha na ligação, rc={rc}")

        def on_disconnect(client, userdata, rc):
            print(f"[{self.client_id}] Desligado (rc={rc}). A tentar reconectar...")

        client = mqtt_client.Client(client_id=self.client_id)
        if user and pwd:
            client.username_pw_set(user, pwd)
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        
        while True:
            try:
                client.connect(broker, port)
                break
            except Exception as e:
                print(f"[{self.client_id}] Broker em baixo... a tentar em 2s.")
                time.sleep(2)
        return client

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, payload, topic):
        full_topic = f"{self.base_topic}{topic}"
        result = self.client.publish(full_topic, json.dumps(payload), qos=self.qos)
        if result[0] != 0:
            print(f"[{self.client_id}] Failed to publish to {full_topic}")

    def on_message(self, client, userdata, msg):
        '''
        Callback para processar mensagens recebidas.
        Acrescenta mensagens à fila interna, ignorando mensagens do próprio client_id.
        '''
        if msg.topic.startswith(f"{self.topic_broker_id}/"): # impede que ouça as suas
            return
        data = json.loads(msg.payload.decode())
        # print(f"[{self.topic_broker_id}] RECEIVED on {msg.topic}: {data}")
        self.msg_queue.put((msg.topic, data)) 

    def subscribe(self, topic):
            if topic not in self.subscribed_topics:
                self.subscribed_topics.append(topic)
            
            if self.client.is_connected():
                self.client.subscribe(topic, qos=self.qos)
