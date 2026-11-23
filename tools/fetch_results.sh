#!/bin/bash
# Traz data_exports/ e results/ de cada Raspberry para experiments/<data>_<nome>/<nó>/ no Mac.
# Nunca apaga nada nos Pis. Modo de uso: docs/deploy.md (secção "Trazer resultados para o Mac")

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

NAME="${1:-run}"
shift || true
SELECTED=("$@")

# Iguais ao tools/deploy.sh / control/node.sh
SSH_OPTS="-F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

python3 tools/gen_configs.py > /dev/null || exit 1

OUT="experiments/$(date +%Y-%m-%d_%H%M)_${NAME}"
mkdir -p "$OUT"

# --- Metadados para reprodutibilidade (que código/config gerou estes dados) ---
cp inventory.yaml "$OUT/"
{
  echo "experiment: ${NAME}"
  echo "fetched_at: $(date -Iseconds)"
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null)"
  echo "git_dirty: $([ -n "$(git status --porcelain 2>/dev/null)" ] && echo yes || echo no)"
} > "$OUT/metadata.yaml"

FAILED=()
while read -r NODE IP USER RPATH; do
  if [ ${#SELECTED[@]} -gt 0 ] && [[ ! " ${SELECTED[*]} " =~ " ${NODE} " ]]; then
    continue
  fi
  echo "📥 ${NODE} (${IP})"
  mkdir -p "$OUT/$NODE"

  for FOLDER in data_exports results; do
    rsync -az -e "ssh $SSH_OPTS" "${USER}@${IP}:${RPATH}/${FOLDER}/" "$OUT/$NODE/$FOLDER/" < /dev/null \
      || { echo "   ⚠️  ${FOLDER}/ não copiado"; FAILED+=("${NODE}:${FOLDER}"); }
  done
done < build/nodes.txt

echo
echo "📁 Resultados em: $OUT"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "⚠️  Falharam: ${FAILED[*]}"
  exit 1
fi
