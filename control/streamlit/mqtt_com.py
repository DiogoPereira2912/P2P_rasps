from datetime import datetime
import json
import paho.mqtt.client as mqtt

class Processor:
    def __init__(
        self,
        raspberry_id,
        broker="",
        port=1883,
        user="admin",
        pwd="public",
        base_topic="system/control/",
        qos=0,
    ):

        self.raspberry_id = raspberry_id
        self.base_topic = base_topic
        self.qos = qos
        self.client = self._connect_mqtt(self.raspberry_id, user, pwd, broker, port)
        self.topic = f"{base_topic}"
        self.subscribed_topics = []
        # self.msg_queue = queue.Queue()
        self.client.loop_start()


    @staticmethod
    def _connect_mqtt(client_id, user, pwd, broker, port):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"Connected to MQTT Broker: {broker}")
            else:
                print(f"Failed to connect, return code {rc}")

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id)
        client.username_pw_set(user, pwd)
        client.on_connect = on_connect
        client.connect(broker, port)
        return client

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
 
    def publish(self, payload, topic):
        result = self.client.publish(
            self.base_topic + topic, json.dumps(payload), qos=self.qos
        )
        status = result[0]
        if status != 0:
            print("Failed to send message")