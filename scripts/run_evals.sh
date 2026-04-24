#!/usr/bin/env bash
# Run both evaluation pipelines.
# Requires OPENROUTER_API_KEY to be set (via .env or environment).
set -euo pipefail

MODEL="openrouter/meta-llama/llama-3.3-70b-instruct"
JUDGE="openrouter/xiaomi/mimo-v2-flash"
ORIGIN="${1:-subsaharan_african}"  # pass as first arg to override
PROVIDER="${2:-}"                   # pass as second arg, e.g. "Fireworks" or "Together"

echo "=== Pipeline 1: Baseline (anonymized) ==="
inspect eval ../inspect_evals/src/inspect_evals/tf_bench/tf_bench.py@tf_bench \
    --model "$MODEL" \
    -T scorer_method=llm \
    -T judge_llm="$JUDGE" \
    -T epochs=1 \
    ${PROVIDER:+-T model_provider="$PROVIDER"} \
    --log-dir logs/baseline

echo ""
echo "=== Pipeline 2: De-anonymized ($ORIGIN) ==="
inspect eval src/scb/tasks/bias_eval.py@bias_tf_bench \
    --model "$MODEL" \
    -T name_origin="$ORIGIN" \
    -T judge_llm="$JUDGE" \
    -T epochs=1 \
    ${PROVIDER:+-T model_provider="$PROVIDER"} \
    --log-dir "logs/bias_${ORIGIN}"
