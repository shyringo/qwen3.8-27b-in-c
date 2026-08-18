#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=Qwen3.8-27B-Q4_K_M.gguf
MODEL_SIZE=17106775008
MODEL_SHA256=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
MODEL_REVISION=d7e4524557ccb88b39f5ab3a925f62f7fec49502
MODEL_DIR=${QWEN38_MODEL_DIR:-"${XDG_CACHE_HOME:-$HOME/.cache}/qwen3.8-27b-in-c/model"}
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
MODELSCOPE_URL="https://www.modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF/resolve/$MODEL_REVISION/$MODEL_NAME"
HUGGINGFACE_URL="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/$MODEL_NAME"

command -v curl >/dev/null 2>&1 || {
    printf "qwen38: curl is required to download the model\n" >&2
    exit 1
}
mkdir -p "$MODEL_DIR"

current_size=0
if [[ -f "$MODEL_PATH" ]]; then
    current_size=$(wc -c < "$MODEL_PATH")
fi
if (( current_size > MODEL_SIZE )); then
    printf "qwen38: %s is larger than the expected model; move it aside and retry\n" \
        "$MODEL_PATH" >&2
    exit 1
fi
downloaded=0
if (( current_size < MODEL_SIZE )); then
    downloaded=1
    printf "Downloading %s from ModelScope (resume enabled)...\n" "$MODEL_NAME"
    if ! curl -fL -C - --retry 8 --retry-delay 3 --connect-timeout 20 \
            -o "$MODEL_PATH" "$MODELSCOPE_URL"; then
        printf "ModelScope download was interrupted; trying Hugging Face...\n" >&2
        curl -fL -C - --retry 8 --retry-delay 3 --connect-timeout 20 \
            -o "$MODEL_PATH" "$HUGGINGFACE_URL"
    fi
fi

current_size=$(wc -c < "$MODEL_PATH")
if [[ "$current_size" != "$MODEL_SIZE" ]]; then
    printf "qwen38: model is %s bytes; expected %s\n" \
        "$current_size" "$MODEL_SIZE" >&2
    exit 1
fi

if (( downloaded )) || [[ ${QWEN38_VERIFY:-0} == 1 ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$MODEL_PATH")
    elif command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 "$MODEL_PATH")
    else
        printf "qwen38: sha256sum or shasum is required to verify the model\n" >&2
        exit 1
    fi
    actual=${actual%% *}
    if [[ "$actual" != "$MODEL_SHA256" ]]; then
        printf "qwen38: model SHA-256 mismatch\n" >&2
        exit 1
    fi
fi

printf "%s\n" "$MODEL_PATH"
