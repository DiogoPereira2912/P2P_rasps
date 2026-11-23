# Seleção de ROIs (regiões de interesse das câmaras)

Corre **no Mac** (pasta `roi_selection/`) porque abre uma janela OpenCV para desenhar as regiões. Tem de ser feito **uma vez por Raspberry** (cada Pi tem as suas câmaras).

## Pré-requisitos
- Os serviços do Pi estão a correr (em particular o `video_server`, porta `8001`).
- No Mac: `pip install opencv-python requests pyyaml`.

## Utilização

```bash
cd roi_selection
python3 ROI_selection_class.py rasp2
```

O nome (`rasp2`) é procurado no `inventory.yaml` para obter o IP. Também aceita um IP ou hostname direto:

```bash
python3 ROI_selection_class.py 192.168.0.111
python3 ROI_selection_class.py hailopi
```

### Passo a passo
1. Abre o feed da **câmara 0** e mostra um frame.
2. Desenhar a ROI com o rato → `ENTER`/`ESPAÇO` para confirmar (`c` cancela).
3. No terminal: `n` para outra ROI, `q` para passar à câmara seguinte.
4. As ROIs são guardadas em `roi_<nó>_video_0.yaml` e enviadas para o Pi.
5. Repete para a **câmara 1** (`roi_<nó>_video_1.yaml`).

No Pi, o `video_server` grava-as em `video_server/rois_url0.yaml` / `rois_url1.yaml` e liberta a câmara. Quando as duas câmaras estão libertadas, o healthcheck do `video_server` passa a OK e os serviços que dependem dele (detector, labeler, broker) arrancam.

### Para os 5 Pis

```bash
for n in rasp1 rasp2 rasp3 rasp4 rasp5; do python3 ROI_selection_class.py $n; done
```

## Configuração — `roi_selection/config.yaml`

```yaml
video_port: 8001   # = WEBPAGE_PORT no .env dos Pis
resolution: { height: 1080, width: 1920 }
```

## ⚠️ Cuidados
- **Repetir a seleção para o mesmo nó acrescenta ROIs ao ficheiro existente** (o `ROI_handler` escreve em modo *append*). Para recomeçar, apagar `roi_<nó>_video_*.yaml` antes.
- **`tools/deploy.sh` / `update_git.sh` substituem** os `video_server/rois_url*.yaml` no Pi pelos do Mac (ver [deploy.md](deploy.md)).
