#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
MODEL_DIR=${QWEN38_MODEL_DIR:-"${XDG_CACHE_HOME:-$HOME/.cache}/qwen3.8-27b-in-c/model"}
MODEL_PATH="$MODEL_DIR/Qwen3.8-27B-Q4_K_M.gguf"
MODEL_SIZE=17106775008
CONTEXT=${QWEN38_CONTEXT:-4096}

if jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null); then
    :
elif command -v sysctl >/dev/null 2>&1; then
    jobs=$(sysctl -n hw.logicalcpu 2>/dev/null || printf "4\n")
else
    jobs=4
fi
case "$jobs" in
    ''|*[!0-9]*|0) jobs=4 ;;
esac
make -s -C "$REPO_DIR" -j"$jobs"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    exec "$REPO_DIR/bin/qwen38" --help
fi
if [[ -f "$MODEL_PATH" ]] && [[ $(wc -c < "$MODEL_PATH") != "$MODEL_SIZE" ]]; then
    "$SCRIPT_DIR/get-model.sh"
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    "$SCRIPT_DIR/get-model.sh"
fi

exec "$REPO_DIR/bin/qwen38" --model "$MODEL_PATH" --context "$CONTEXT" "$@"
