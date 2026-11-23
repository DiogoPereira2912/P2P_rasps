#!/bin/bash
# Envia código + config gerada (inventory.yaml) para os Raspberries.
# Modo de uso: docs/deploy.md (secção "Enviar código + config")

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RESTART=false
DRY=""
SELECTED=()
for arg in "$@"; do
  case "$arg" in
    --restart) RESTART=true ;;
    --dry-run) DRY="-n" ;;
    *) SELECTED+=("$arg") ;;
  esac
done

PY_SERVICES="peer pipeline aggregation labeler detector"

# Iguais ao control/node.sh: -F /dev/null ignora o ~/.ssh/config do Mac (opções como
# UseKeychain não existem no ssh do contentor da UI) e usa as chaves de ~/.ssh.
SSH_OPTS="-F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

python3 tools/gen_configs.py || exit 1
echo

FAILED=()
while read -r NAME IP USER RPATH; do
  if [ ${#SELECTED[@]} -gt 0 ] && [[ ! " ${SELECTED[*]} " =~ " ${NAME} " ]]; then
    continue
  fi

  echo "====================================================="
  echo "🚀 ${NAME} (${IP})"
  echo "====================================================="

  # "< /dev/null": impede o rsync/ssh de consumir o resto do nodes.txt (stdin do ciclo)
  if ! rsync -az $DRY -e "ssh $SSH_OPTS" --delete --exclude-from=.rsyncignore \
       ./ "${USER}@${IP}:${RPATH}/" < /dev/null; then
    echo "❌ ${NAME}: falha a enviar o código"; FAILED+=("$NAME"); continue
  fi

  if ! rsync -az $DRY -e "ssh $SSH_OPTS" "build/nodes/${NAME}/" "${USER}@${IP}:${RPATH}/" < /dev/null; then
    echo "❌ ${NAME}: falha a enviar a config"; FAILED+=("$NAME"); continue
  fi

  if $RESTART && [ -z "$DRY" ]; then
    ssh -n $SSH_OPTS "${USER}@${IP}" "cd ${RPATH} && docker compose restart ${PY_SERVICES}" < /dev/null \
      || { echo "❌ ${NAME}: falha no restart"; FAILED+=("$NAME"); continue; }
  fi

  echo "✅ ${NAME} atualizado"
done < build/nodes.txt

echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "⚠️  Falharam: ${FAILED[*]}"
  exit 1
fi
echo "✅ Deploy concluído."
