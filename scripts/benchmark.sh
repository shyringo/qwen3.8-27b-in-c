#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROMPT=${QWEN38_BENCH_PROMPT:-"科技的边界在哪里？"}
TOKENS=${QWEN38_BENCH_TOKENS:-32}

case "$TOKENS" in
    ''|*[!0-9]*|0)
        printf "qwen38: QWEN38_BENCH_TOKENS must be a positive integer\n" >&2
        exit 2
        ;;
esac

exec "$SCRIPT_DIR/chat.sh" \
    --prompt "$PROMPT" \
    --no-thinking \
    --temperature 0 \
    --presence-penalty 0 \
    --max-tokens "$TOKENS" \
    "$@"
