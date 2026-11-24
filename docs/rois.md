# Seleção de ROIs (regiões de interesse das câmaras)

Corre **no Mac** (pasta `roi_selection/`) porque abre uma janela OpenCV para desenhar as regiões. Tem de ser feito **uma vez por Raspberry** (cada Pi tem as suas câmaras).

## Pré-requisitos
- Os serviços do Pi estão a correr (em particular o `video_server`, porta `8001`).
- No Mac: `pip install opencv-python requests pyyaml`.

## Utilização

**Todos os Pis seguidos** (o mais rápido):
```bash
cd roi_selection
./select_rois.sh                 # todos os nós do inventory.yaml
./select_rois.sh rasp2 rasp4     # só alguns
```

**Um Pi** (nome do inventory, IP ou hostname):
```bash
python3 ROI_selection_class.py rasp2
```

### Passo a passo (por Pi)
1. Abre um frame da **câmara 0**.
2. Desenhar a ROI com o rato → `ENTER`/`ESPAÇO` confirma (`c` cancela).
3. No terminal: `n` para outra ROI, `q` para terminar.
4. As ROIs são guardadas em `roi_<nó>_video_0.yaml` (o anterior é apagado) e enviadas para o Pi.

As ROIs são as áreas onde o modelo deteta e classifica pessoas: uma, várias ou a imagem toda.
Cuidados para os artigos:
- **Não sobrepor ROIs** na mesma câmara (uma pessoa na sobreposição é detetada duas vezes).
- **O YOLO corre uma vez por ROI**: manter a mesma configuração de ROIs em todos os runs
  (e registar no artigo quantas ROIs tem cada câmara).

Só a câmara 0 é usada pelo `video_server` e pelo detetor. Para duas câmaras: `--duas-cameras`.

No Pi, o `video_server` grava-as em `video_server/rois_url0.yaml` e liberta a câmara; o
healthcheck passa a OK e os serviços que dependem dele (detector, labeler, broker) arrancam.

## Configuração — `roi_selection/config.yaml`

```yaml
video_port: 8001   # = WEBPAGE_PORT no .env dos Pis
resolution: { height: 1080, width: 1920 }
```

## ⚠️ Cuidados
- As ROIs dos Pis (`video_server/rois_url*.yaml`) estão no `.rsyncignore`: o `tools/deploy.sh`
  já **não** as substitui pelas do Mac.
