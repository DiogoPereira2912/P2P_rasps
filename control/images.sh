#!/bin/bash
# Cria os .tar das imagens Docker (arm64) em control/imgs/ para deploy offline.
# Modo de uso: docs/deploy.md (secção "Imagens Docker")

cd "$(dirname "$0")" || exit 1

ACTION=$1
PLATFORM="linux/arm64"   # Raspberry Pi 5 (64 bits)
IMGS_DIR="imgs"

mkdir -p "$IMGS_DIR"

case "$ACTION" in

  build)
    FOLDER_PATH=$2
    IMAGE_NAME=$3
    if [ -z "$FOLDER_PATH" ] || [ -z "$IMAGE_NAME" ]; then
      echo "👉 Uso: ./images.sh build <pasta_dockerfile> <nome_imagem>"
      exit 1
    fi
    if [ ! -d "$FOLDER_PATH" ]; then
      echo "❌ Erro: A pasta '$FOLDER_PATH' não existe."
      exit 1
    fi

    echo "📦 1/2 - A compilar ${IMAGE_NAME}:latest para ${PLATFORM}..."
    docker buildx build --platform $PLATFORM -t "${IMAGE_NAME}:latest" --load "$FOLDER_PATH" || {
      echo "❌ Erro no build da imagem."; exit 1; }

    echo "💾 2/2 - A exportar para ${IMGS_DIR}/${IMAGE_NAME}.tar..."
    docker save -o "${IMGS_DIR}/${IMAGE_NAME}.tar" "${IMAGE_NAME}:latest" || {
      echo "❌ Erro ao exportar a imagem."; exit 1; }
    ;;

  pull)
    IMAGE=$2
    TAR_NAME=$3
    if [ -z "$IMAGE" ] || [ -z "$TAR_NAME" ]; then
      echo "👉 Uso: ./images.sh pull <imagem:tag> <nome_tar>"
      exit 1
    fi

    echo "⬇️  1/2 - A descarregar ${IMAGE} (${PLATFORM})..."
    docker pull --platform $PLATFORM "$IMAGE" || {
      echo "❌ Erro no pull da imagem."; exit 1; }

    echo "💾 2/2 - A exportar para ${IMGS_DIR}/${TAR_NAME}.tar..."
    docker save -o "${IMGS_DIR}/${TAR_NAME}.tar" "$IMAGE" || {
      echo "❌ Erro ao exportar a imagem."; exit 1; }
    ;;

  *)
    echo "👉 Uso: ./images.sh build <pasta_dockerfile> <nome_imagem>"
    echo "        ./images.sh pull  <imagem:tag> <nome_tar>"
    exit 1
    ;;
esac

echo "✅ Concluído! Ficheiro pronto em ${IMGS_DIR}/ para deploy."
