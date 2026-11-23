import cv2, requests, os, sys
from ROI_handler import ROIHandler
from yaml import Loader, load

# inventory.yaml está na raiz do repo (um nível acima desta pasta)
INVENTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inventory.yaml")


def resolve_host(target):
    """
    Converte o nome de um nó (ex: "rasp2") no IP definido no inventory.yaml.
    Se o nome não existir no inventário, usa o valor tal como está
    (permite passar diretamente um IP ou hostname, ex: "hailopi").
    """
    if os.path.exists(INVENTORY_PATH):
        with open(INVENTORY_PATH, "r") as file:
            inventory = load(file, Loader=Loader)
        node = inventory["nodes"].get(target)
        if node:
            return node["ip"]
    return target


class ROIManager:

    def __init__(self, target, config_path="config.yaml"):
        """
        target: nome do nó no inventory.yaml (ex: "rasp2"), IP ou hostname
        config_path: config local (porta do video server, resolução)
        """
        with open(config_path, "r") as file:
            self.config = load(file, Loader=Loader)

        host = resolve_host(target)
        self.base_url = f"http://{host}:{self.config['video_port']}"
        # Feeds das câmaras expostos pelo video_server no Pi
        self.video_urls = {
            "url0": f"{self.base_url}/video/0",
            "url1": f"{self.base_url}/video/1",
        }
        print(f"🎯 Nó alvo: {target} -> {self.base_url}")

    def select_rois_for_video(self, video_key, yaml_path):
        """
        Abre o feed das cameras para seleção de ROIs através da url -> video_key
        Guarda as coordenadas das regiões selecionadas num ficheiro YAML -> yaml_path
        """

        url = self.video_urls[video_key]

        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir a captura de vídeo: {url}")
            return []

        ret, frame = cap.read()
        if not ret:
            print(f"Erro: Não foi possível ler o frame de: {url}")
            cap.release()
            return []

        roi_handlers = []
        count = 1

        frame_para_selecao = frame.copy()

        while True:
            roi_handler = ROIHandler()
            roi = roi_handler.select_and_save_roi(frame_para_selecao, count, yaml_path)

            if roi:
                roi_handlers.append(roi_handler)
                print(f"ROI {count} selecionado: {roi}")
                count += 1
            else:
                print("Seleção de ROI cancelada.")

            key = (
                input("Pressione 'n' para outro ROI ou 'q' para terminar: ")
                .strip()
                .lower()
            )
            if key == "q":
                break

        cap.release()
        cv2.destroyAllWindows()

        return roi_handlers

    def send_roi(self, video_key, local_yaml_path):
        """
        Função 'helper' para ler um ficheiro YAML local e enviá-lo para o Pi.
        """

        if not os.path.exists(local_yaml_path):
            print(
                f"Erro no PC: Ficheiro {local_yaml_path} não encontrado. Envio cancelado."
            )
            return False

        url = f"{self.base_url}/upload_roi"

        params = {"video_key": video_key}

        with open(local_yaml_path, "r") as f:
            yaml_content = f.read()

        print(f"A enviar {local_yaml_path} para o Pi em {url}...")

        response = requests.post(
            url,
            params=params,
            data=yaml_content,
            headers={"Content-Type": "text/plain"}
        )

        if response.status_code == 200:
            print(response.json())
            return True
        else:
            print(response.text)
            return False


if __name__ == "__main__":
    # Modo de uso: docs/rois.md
    if len(sys.argv) < 2:
        print("👉 Uso: python3 ROI_selection_class.py <nome_do_nó | ip | hostname>")
        sys.exit(1)

    target = sys.argv[1]
    manager = ROIManager(target)

    # Um ficheiro por nó e por câmara, para não misturar as ROIs de Pis diferentes
    yaml_video_0 = f"roi_{target}_video_0.yaml"
    yaml_video_1 = f"roi_{target}_video_1.yaml"

    rois_video_1 = manager.select_rois_for_video(video_key="url0", yaml_path=yaml_video_0)
    manager.send_roi(video_key="url0", local_yaml_path=yaml_video_0)

    rois_video_2 = manager.select_rois_for_video(video_key="url1", yaml_path=yaml_video_1)
    manager.send_roi(video_key="url1", local_yaml_path=yaml_video_1)