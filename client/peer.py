import socket
import threading
import time
import os
from yaml import Loader, load
from mqtt_layer import Communication_Layer


class Peer:

    def __init__(self):

        with open("config.yaml", "r") as file:
            self.config = load(file, Loader=Loader)

        self.mosquitto_port = self.config["mosquitto_port"]
        self.broadcast_port = self.config["broadcast_port"]
        self.broadcast_mask = self.config["broadcast_mask"]
        self.peer_ip = self.config["peer_ip"]
        self.broker_id = self.peer_ip.replace(".", "_")
        self.prefix = f"br_{self.broker_id.split('_')[-1]}/"
        self.known_peers = []
        # --- inicialização de cliente MQTT e socket --- #
        self._setup_mqtt_client()
        self._setup_socket()
        self._start_listener()
        self._notify_peers()

    def _setup_mqtt_client(self):
        """
        Cria o cliente MQTT e faz o subscribe ao tópico
        """
        try:
            self.mqtt_com = Communication_Layer(
                broker=self.peer_ip,
                port=self.mosquitto_port,
                client_id=f"{self.broker_id}",
                base_topic="",
                qos=1,
            )
            self.mqtt_com.subscribe(topic="#")
        except Exception as e:
            print(f"Erro ao conectar ao broker MQTT: {e}")

    def _reload_mqtt_configs(self):
        """
        Reinicia o Mosquitto e reconecta o cliente MQTT.
        Reload das configurações das bridges.
        """
        self.mqtt_com.disconnect()
        os.system("docker restart mqtt_broker")
        self._setup_mqtt_client()

    def _setup_socket(self):
        """
        Função de criação do socket para a descoberta de outros peers.
        Método apenas usado por esta classe (_) no inicio do nome da função.
        """
        self.sock_broadcast = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_broadcast.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock_broadcast.bind(("", self.broadcast_port))

    def _start_listener(self):
        """
        Iniciar a thread de listener para a descoberta de outros peers
        """
        listener = threading.Thread(target=self._listen_for_peers, daemon=True)
        listener.start()

    def _notify_peers(self):
        """
        Troca de informçoes entre peers, envia a porta do broker
        """
        self.sock_broadcast.sendto(
            f"hello {self.mosquitto_port}".encode(),
            (self.broadcast_mask, self.broadcast_port),
        )

    def add_mosquitto_bridge(self, remote_peer_ip, remote_peer_port):
        """
        Adiciona uma bridge Mosquitto para um peer recém-descoberto.
        Usa o peer IP como broker_id para o tópico remoto.

        Args:
            remote_peer_ip (str): IP do peer remoto
            remote_peer_port (int): Porta MQTT do peer remoto
        """
        remote_peer_id = remote_peer_ip.replace(".", "_")

        with open(
            "./mosquitto/conf.d/bridges.conf", "r"
        ) as f:  # read the bridges.conf file
            existing_conns = [line.strip() for line in f]

        current_address = f"address {remote_peer_ip}:{remote_peer_port}"

        if any(
            current_address in line for line in existing_conns
        ):  # check if the address has a bridge conn already
            pass
        else:
            bridge_config = f"""
    connection bridge_to_peer__{remote_peer_id}
    address {remote_peer_ip}:{remote_peer_port}
    topic {self.broker_id}/# out 1 {self.prefix}
    topic {remote_peer_id}/# in 1
                """

            with open("./mosquitto/conf.d/bridges.conf", "a") as f:
                f.write(f"\n{bridge_config}")

            self._reload_mqtt_configs()

    def _listen_for_peers(self):
        """
        Thread que escuta mensagens de broadcast para descoberta de peers.
        - Recebe "hello" ou "hi" com porta MQTT
        - Responde para informar seu próprio broker
        """
        while True:
            data, addr = self.sock_broadcast.recvfrom(1024)
            msg = data.decode()

            if addr[0] == self.peer_ip:
                continue

            if msg.startswith("hello") or msg.startswith("hi"):
                _, peer_mqtt_port = msg.split()
                peer_mqtt_port = int(peer_mqtt_port)
                peer = (addr[0], peer_mqtt_port)

                if peer not in self.known_peers:
                    self.known_peers.append(peer)
                    print(f"Peers encontrados: {self.known_peers}")
                    for p in self.known_peers:
                        self.add_mosquitto_bridge(p[0], p[1])
                        if p != peer:
                            self.sock_broadcast.sendto(f"hi {peer[1]}".encode(), p)

                if msg.startswith("hello"):
                    self.sock_broadcast.sendto(
                        f"hi {self.mosquitto_port}".encode(), addr
                    )

    def data_publisher(self):
        ts = 1
        while True:
            time.sleep(3)
            data_sample = {"ts": ts, "msg": f"Cheguei #{ts} de {self.peer_ip}"}
            try:
                self.mqtt_com.publish(data_sample, f"{self.broker_id}/test")
            except Exception as e:
                print(f"Erro ao publicar: {e}")

            print(f"Enviado")
            ts += 1


if __name__ == "__main__":
    peer = Peer()
    peer.data_publisher()