#!/usr/bin/env bash
set -euo pipefail

QUANT=${QWEN38_QUANT:-iq1_m}
case "$QUANT" in
    iq1_m)
        MODEL_NAME=Qwen3.8-27B-UD-IQ1_M.gguf
        MODEL_SIZE=6729166848
        MODEL_SHA256=1b5165a7149ea51e683c8eaf23372188ad9fc9d1a795386f7a1b558acf847dc6
        MODEL_REVISION=1fa4a98544ac96043a10649853051f1d5e72a008
        ;;
    q4_k_m)
        MODEL_NAME=Qwen3.8-27B-UD-Q4_K_M.gguf
        MODEL_SIZE=16464440224
        MODEL_SHA256=322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482
        MODEL_REVISION=ba7608d4e5e1f3ea3d016cebd1c972c42686e9da
        ;;
    iq2_m)
        MODEL_NAME=Qwen3.8-27B-UD-IQ2_M.gguf
        MODEL_SIZE=10319907904
        MODEL_SHA256=04a89ef4fa9c8726d09331433346809bbab692b4851d49d0738ba8d58a1ae740
        MODEL_REVISION=1939530128ebc56fdd0213d25583cb85147c1cd5
        ;;
    *)
        printf "qwen38: QWEN38_QUANT must be iq1_m, iq2_m, or q4_k_m\n" >&2
        exit 2
        ;;
esac
MODEL_DIR=${QWEN38_MODEL_DIR:-"${XDG_CACHE_HOME:-$HOME/.cache}/qwen3.8-27b-in-c/model"}
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
VERIFIED_MARKER="$MODEL_PATH.$MODEL_SHA256.ok"
FREE_SPACE_RESERVE=$((1024 * 1024 * 1024))
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
    printf "  mv -- %q %q\n" "$MODEL_PATH" "$MODEL_PATH.unexpected" >&2
    exit 1
fi
downloaded=0
if (( current_size < MODEL_SIZE )); then
    remaining=$((MODEL_SIZE - current_size))
    available_kib=$(df -Pk "$MODEL_DIR" 2>/dev/null | awk 'END { print $4 }')
    case "$available_kib" in
        ''|*[!0-9]*) ;;
        *)
            if (( available_kib * 1024 < remaining + FREE_SPACE_RESERVE )); then
                required_gib=$(awk -v bytes="$remaining" \
                    'BEGIN { printf "%.1f", bytes / 1073741824 }')
                available_gib=$(awk -v kib="$available_kib" \
                    'BEGIN { printf "%.1f", kib / 1048576 }')
                printf "qwen38: the download needs %s GiB more plus 1 GiB free-space reserve, but %s has only %s GiB free\n" \
                    "$required_gib" "$MODEL_DIR" "$available_gib" >&2
                exit 1
            fi
            ;;
    esac
    downloaded=1
    printf "Downloading %s from ModelScope (resume enabled)...\n" \
        "$MODEL_NAME" >&2
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

if (( downloaded )) || [[ ${QWEN38_VERIFY:-0} == 1 ]] ||
        [[ ! -f "$VERIFIED_MARKER" ]] || [[ "$MODEL_PATH" -nt "$VERIFIED_MARKER" ]]; then
    printf "Verifying %s...\n" "$MODEL_NAME" >&2
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
        printf "qwen38: model SHA-256 mismatch; move %s aside and retry\n" \
            "$MODEL_PATH" >&2
        printf "  mv -- %q %q\n" "$MODEL_PATH" "$MODEL_PATH.corrupt" >&2
        exit 1
    fi
    touch "$VERIFIED_MARKER"
fi

printf "%s\n" "$MODEL_PATH"
