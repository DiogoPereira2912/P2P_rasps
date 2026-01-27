import socket, threading, time, os, json
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
        self.node_id = self.config["node_id"]
        self.known_peers = []
        self.device_id = self.config["device_id"]
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
        time.sleep(1)
        os.system("docker restart mqtt_broker")
        time.sleep(2)
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
            Troca de informçoes entre peers, envia a porta, node_id E O DEVICE_ID
            """
            msg = f"hello {self.mosquitto_port} {self.node_id} {self.device_id}"
            self.sock_broadcast.sendto(
                msg.encode(),
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
            topic {self.broker_id}/# out 1 
            topic {remote_peer_id}/# in 1
            topic {remote_peer_id}/# out 1
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
                _, peer_mqtt_port, peer_network_id, peer_device_id = msg.split()
                peer_mqtt_port = int(peer_mqtt_port)
                peer_id = int(peer_network_id)
                peer = (addr[0], peer_mqtt_port, peer_id, peer_device_id)

                if peer not in self.known_peers:
                    self.known_peers.append(peer)
                    print(f"Peers encontrados: {self.known_peers}")
                    for p in self.known_peers:
                        self.add_mosquitto_bridge(p[0], p[1])
                        if p != peer:
                            self.sock_broadcast.sendto(f"hi {p[1]} {p[2]} {p[3]}".encode(), (p[0], p[1]))
                    self._broadcast_known_peers_internally()

                if msg.startswith("hello"):
                    self.sock_broadcast.sendto(
                        f"hi {self.mosquitto_port} {self.node_id} {self.device_id}".encode(), addr
                    )

    def _broadcast_known_peers_internally(self): 
        """ 
        Envia a lista de IPs para os outros módulos (Pipeline/Agg) usarem.
        """
        ip_list = sorted([(p[0], p[2], p[3]) for p in self.known_peers])
        known_peers_payload = json.dumps(ip_list)
        self.mqtt_com.client.publish(payload = known_peers_payload, topic = "system/peers", retain=True)

if __name__ == "__main__":
    peer = Peer() 
    try:
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        print("\nA desligar o Peer...")