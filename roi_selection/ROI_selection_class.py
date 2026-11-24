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
    # Modo de uso: docs/rois.md (ou ./select_rois.sh para todos os Pis)
    if len(sys.argv) < 2:
        print("👉 Uso: python3 ROI_selection_class.py <nome_do_nó | ip | hostname> [--duas-cameras]")
        sys.exit(1)

    target = sys.argv[1]
    # O video_server só usa a câmara 0 (CAM1 comentada): a câmara 1 só com --duas-cameras
    cameras = ["url0", "url1"] if "--duas-cameras" in sys.argv else ["url0"]
    manager = ROIManager(target)

    for video_key in cameras:
        # Um ficheiro por nó e por câmara. Apagar o anterior: o ROIHandler escreve em
        # modo append e, sem isto, as ROIs novas juntavam-se às antigas.
        yaml_path = f"roi_{target}_video_{video_key[-1]}.yaml"
        if os.path.exists(yaml_path):
            os.remove(yaml_path)
        manager.select_rois_for_video(video_key=video_key, yaml_path=yaml_path)
        manager.send_roi(video_key=video_key, local_yaml_path=yaml_path)
