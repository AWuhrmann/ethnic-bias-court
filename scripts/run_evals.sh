#!/usr/bin/env bash
# Run evaluation pipelines.
#
# Usage:
#   ./run_evals.sh                        # baseline + all origins
#   ./run_evals.sh --baseline             # baseline only
#   ./run_evals.sh --origin subsaharan_african   # one origin only (no baseline)
#   ./run_evals.sh --origin subsaharan_african --baseline  # one origin + baseline
#
# Optional flags:
#   --model <model>       override MODEL
#   --judge <model>       override JUDGE
#   --provider <name>     e.g. "Fireworks"
#   --max-connections N   default: 70
#   --epochs N            default: 3
#   --names-file path     custom names yaml (default: data/names/groups.yaml)
set -euo pipefail

MODEL="vllm/meta-llama/llama-3.3-70b-instruct"
JUDGE="openrouter/xiaomi/mimo-v2-flash"
PROVIDER=""
MAX_CONNECTIONS=70
EPOCHS=3
ORIGIN=""
NAMES_FILE=""
RUN_BASELINE=false
RUN_ALL_ORIGINS=false
VLLM_BASE_URL="http://localhost:8000/v1"

ALL_ORIGINS=(
    swiss_german
    swiss_french
    swiss_italian
    turkish
    albanian
    serbian
    arab_maghrebi
    subsaharan_african
    south_asian
)

# --- argument parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)    RUN_BASELINE=true ;;
        --all-origins) RUN_ALL_ORIGINS=true ;;
        --origin)      ORIGIN="$2"; shift ;;
        --model)      MODEL="$2"; shift ;;
        --judge)      JUDGE="$2"; shift ;;
        --provider)   PROVIDER="$2"; shift ;;
        --max-connections) MAX_CONNECTIONS="$2"; shift ;;
        --epochs)      EPOCHS="$2"; shift ;;
        --names-file)     NAMES_FILE="$2"; shift ;;
        --vllm-base-url)  VLLM_BASE_URL="$2"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

# Default: no explicit flags → run everything
if [[ "$RUN_BASELINE" == false && "$RUN_ALL_ORIGINS" == false && -z "$ORIGIN" ]]; then
    RUN_BASELINE=true
    ORIGINS_TO_RUN=("${ALL_ORIGINS[@]}")
elif [[ "$RUN_ALL_ORIGINS" == true ]]; then
    ORIGINS_TO_RUN=("${ALL_ORIGINS[@]}")
elif [[ -n "$ORIGIN" ]]; then
    ORIGINS_TO_RUN=("$ORIGIN")
else
    ORIGINS_TO_RUN=()
fi

PROVIDER_FLAG=""
[[ -n "$PROVIDER" ]] && PROVIDER_FLAG="-T model_provider=$PROVIDER"
NAMES_FILE_FLAG=""
[[ -n "$NAMES_FILE" ]] && NAMES_FILE_FLAG="-T names_file=$NAMES_FILE"

# --- baseline ---
if [[ "$RUN_BASELINE" == true ]]; then
    echo "=== Pipeline 1: Baseline (anonymized) ==="
    inspect eval ../inspect_evals/src/inspect_evals/tf_bench/tf_bench.py@tf_bench \
        --model "$MODEL" \
        --max-connections "$MAX_CONNECTIONS" \
        ${VLLM_BASE_URL:+--model-base-url "$VLLM_BASE_URL"} \
        -T scorer_method=llm \
        -T judge_llm="$JUDGE" \
        -T epochs="$EPOCHS" \
        ${PROVIDER_FLAG:+$PROVIDER_FLAG} \
        --log-dir logs/baseline
fi

# --- bias evals ---
for ORIG in "${ORIGINS_TO_RUN[@]}"; do
    echo ""
    echo "=== Pipeline 2: De-anonymized ($ORIG) ==="
    inspect eval src/scb/tasks/bias_eval.py@bias_tf_bench \
        --model "$MODEL" \
        --max-connections "$MAX_CONNECTIONS" \
        ${VLLM_BASE_URL:+--model-base-url "$VLLM_BASE_URL"} \
        -T name_origin="$ORIG" \
        -T judge_llm="$JUDGE" \
        -T epochs="$EPOCHS" \
        ${PROVIDER_FLAG:+$PROVIDER_FLAG} \
        ${NAMES_FILE_FLAG:+$NAMES_FILE_FLAG} \
        --log-dir "logs/bias_${ORIG}"
done
