#!/bin/bash
# Ações SSH num Raspberry: 'start' (reiniciar serviços) e 'load' (carregar imagem .tar).
# Chamado pelos botões da UI. Modo de uso: docs/deploy.md (secção "Operações num nó")

ACTION=$1
TARGET_IP=$2

# Têm de coincidir com ssh_user / remote_path do inventory.yaml
RASP_USER="${RASP_USER:-aidi}"
RASP_DEST_DIR="${RASP_DEST_DIR:-/home/aidi/P2P}"

# -F /dev/null ignora o ~/.ssh/config do Mac (tem opções como UseKeychain que o
# ssh do contentor não conhece) e usa as chaves de ~/.ssh diretamente.
SSH_OPTS="-F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

if [ -z "$TARGET_IP" ]; then
  echo "👉 Uso: ./node.sh start <ip>"
  echo "        ./node.sh load  <ip> <ficheiro.tar>"
  exit 1
fi

case "$ACTION" in

  start)
    echo "🔌 A reiniciar os serviços em $TARGET_IP..."
    ssh $SSH_OPTS "${RASP_USER}@${TARGET_IP}" \
      "cd ${RASP_DEST_DIR} && docker compose down && docker compose up -d" || {
      echo "❌ Erro ao iniciar os serviços em $TARGET_IP."; exit 1; }
    ;;

  load)
    TAR_FILE_PATH=$3
    if [ -z "$TAR_FILE_PATH" ]; then
      echo "❌ Erro: Falta o ficheiro .tar."
      exit 1
    fi
    TAR_BASENAME=$(basename "$TAR_FILE_PATH")

    echo "🚀 1/2 - A transferir ${TAR_BASENAME} para $TARGET_IP..."
    rsync -e "ssh $SSH_OPTS" -P "$TAR_FILE_PATH" "${RASP_USER}@${TARGET_IP}:${RASP_DEST_DIR}/" || {
      echo "❌ Erro na transferência para $TARGET_IP."; exit 1; }

    echo "⚙️ 2/2 - A carregar a imagem no Docker do Pi..."
    ssh $SSH_OPTS "${RASP_USER}@${TARGET_IP}" \
      "cd ${RASP_DEST_DIR} && docker load -i ${TAR_BASENAME} && rm ${TAR_BASENAME}" || {
      echo "❌ Erro ao carregar a imagem em $TARGET_IP."; exit 1; }
    ;;

  *)
    echo "❌ Ação desconhecida: '$ACTION' (usar 'start' ou 'load')"
    exit 1
    ;;
esac

echo "✅ Concluído em $TARGET_IP!"
