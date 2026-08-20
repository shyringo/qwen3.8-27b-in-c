#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONTEXT=${QWEN38_CONTEXT:-4096}

export OMP_DYNAMIC=${OMP_DYNAMIC:-false}

# Keep workers active only for a process that exits after one request. During
# an interactive session, sleeping workers avoid consuming CPU between turns.
if [[ -z ${OMP_WAIT_POLICY:-} ]]; then
    one_shot=0
    for argument in "$@"; do
        case "$argument" in
            --prompt|--prompt-file) one_shot=1 ;;
        esac
    done
    if (( one_shot )); then
        export OMP_WAIT_POLICY=ACTIVE
    else
        export OMP_WAIT_POLICY=PASSIVE
    fi
fi

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

# WSL2 exposes hybrid laptop CPUs through a synthetic topology. Keeping the
# default 12 workers on a stable vCPU set avoids costly migrations while still
# allowing explicit OpenMP settings to take precedence.
if [[ -z ${QWEN38_THREADS:-} && -z ${GOMP_CPU_AFFINITY:-} &&
      -z ${OMP_PROC_BIND:-} && $jobs -gt 12 &&
      -r /proc/version ]] && grep -qi microsoft /proc/version; then
    export GOMP_CPU_AFFINITY=0-11
fi

make -s -C "$REPO_DIR" -j"$jobs"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    exec "$REPO_DIR/bin/qwen38" --help
fi

explicit_model=0
for argument in "$@"; do
    if [[ "$argument" == --model ]]; then
        explicit_model=1
        break
    fi
done

if (( explicit_model )); then
    exec "$REPO_DIR/bin/qwen38" --context "$CONTEXT" "$@"
elif [[ -n ${QWEN38_MODEL:-} ]]; then
    MODEL_PATH=$QWEN38_MODEL
    if [[ ! -f "$MODEL_PATH" ]]; then
        printf "qwen38: QWEN38_MODEL is not a file: %s\n" "$MODEL_PATH" >&2
        exit 1
    fi
    if [[ ! -r "$MODEL_PATH" ]]; then
        printf "qwen38: QWEN38_MODEL is not readable: %s\n" "$MODEL_PATH" >&2
        exit 1
    fi
else
    if [[ -z ${QWEN38_QUANT:-} ]]; then
        memory_kib=0
        if [[ -r /proc/meminfo ]]; then
            memory_kib=$(awk '
                $1 == "MemAvailable:" { available = $2 }
                $1 == "MemTotal:" { total = $2 }
                END { print available ? available : total }
            ' /proc/meminfo)
        elif command -v sysctl >/dev/null 2>&1; then
            memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || printf "0\n")
            case "$memory_bytes" in
                ''|*[!0-9]*) memory_bytes=0 ;;
            esac
            if command -v vm_stat >/dev/null 2>&1; then
                page_size=$(sysctl -n hw.pagesize 2>/dev/null || printf "4096\n")
                case "$page_size" in
                    ''|*[!0-9]*|0) page_size=4096 ;;
                esac
                available_pages=$(vm_stat | awk '
                    /Pages free:/ { gsub("\\.", "", $3); free = $3 }
                    /Pages inactive:/ { gsub("\\.", "", $3); inactive = $3 }
                    /Pages speculative:/ { gsub("\\.", "", $3); speculative = $3 }
                    END { print free + inactive + speculative }
                ')
                case "$available_pages" in
                    ''|*[!0-9]*) available_pages=0 ;;
                esac
                if (( available_pages > 0 )); then
                    memory_bytes=$((available_pages * page_size))
                fi
            fi
            memory_kib=$((memory_bytes / 1024))
        fi
        case "$memory_kib" in
            ''|*[!0-9]*) memory_kib=0 ;;
        esac
        if (( memory_kib >= 20 * 1024 * 1024 )); then
            QWEN38_QUANT=q4_k_m
            printf "qwen38: selecting Dynamic V3 Q4_K_M weights for available memory\n" >&2
        else
            QWEN38_QUANT=iq1_m
            printf "qwen38: selecting the 8 GB Dynamic V3 IQ1_M path\n" >&2
        fi
        export QWEN38_QUANT
    fi
    MODEL_PATH=$("$SCRIPT_DIR/get-model.sh")
fi

exec "$REPO_DIR/bin/qwen38" --model "$MODEL_PATH" --context "$CONTEXT" "$@"
