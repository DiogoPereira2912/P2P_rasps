#!/bin/bash
# Seleção de ROIs em vários Pis seguidos (câmara 0). Modo de uso: docs/rois.md
#   ./select_rois.sh                 -> todos os nós do inventory.yaml
#   ./select_rois.sh rasp2 rasp4     -> só estes
#   ./select_rois.sh --duas-cameras  -> câmaras 0 e 1 (se o video_server usar as duas)

cd "$(dirname "$0")" || exit 1

EXTRA=""
NODES=()
for arg in "$@"; do
  if [ "$arg" = "--duas-cameras" ]; then EXTRA="--duas-cameras"; else NODES+=("$arg"); fi
done

if [ ${#NODES[@]} -eq 0 ]; then
  NODES=($(python3 -c "import yaml; print(' '.join(yaml.safe_load(open('../inventory.yaml'))['nodes']))"))
fi

for NODE in "${NODES[@]}"; do
  echo "====================================================="
  echo "🎯 ${NODE}: desenhar as ROIs (ENTER confirma cada uma), depois 'n' outra ROI / 'q' terminar"
  echo "====================================================="
  python3 ROI_selection_class.py "$NODE" $EXTRA || echo "⚠️  ${NODE}: falhou (video_server a correr? porta 8001)"
done

echo "✅ ROIs enviadas. Confirmar no Pi: ls ~/P2P/video_server/rois_url0.yaml"
