#!/usr/bin/env bash
# Step 3: Run Qwen3.5 models on benchmarks.
#
# App names (from serve_qwen35_instruct.py):
#   agentflow-qwen35-08b  (0.8B, A10G)
#   agentflow-qwen35-2b   (2B,   A10G)
#   agentflow-qwen35-4b   (4B,   A10G)
#   agentflow-qwen35-9b   (9B,   A100)
#
# Usage:
#   bash scripts/run_step3.sh <model_size> <planner_base_url>
#
# Example:
#   bash scripts/run_step3.sh 0.8b "https://varunkasa77--agentflow-qwen35-08b-serve.modal.run/v1"
#   bash scripts/run_step3.sh 4b   "https://varunkasa77--agentflow-qwen35-4b-serve.modal.run/v1"
#   bash scripts/run_step3.sh 9b   "https://varunkasa77--agentflow-qwen35-9b-serve.modal.run/v1"

MODEL_SIZE=${1:-"0.8b"}
PLANNER_URL=${2}
PYTHON="python"
EXP_NAME="step3_qwen35_${MODEL_SIZE//.}"
# Qwen3.5 has NO -Instruct suffix; model is served as "Qwen/Qwen3.5-{SIZE}"
MODEL_STR="vllm-Qwen/Qwen3.5-${MODEL_SIZE^^}"

if [ -z "$PLANNER_URL" ]; then
    echo "Usage: bash scripts/run_step3.sh <model_size> <planner_base_url>"
    exit 1
fi

echo "=== Step 3: Qwen3.5-${MODEL_SIZE} ==="
echo "Planner URL: $PLANNER_URL"
echo "Exp name: $EXP_NAME"

BENCHMARKS=${3:-"bamboogle"}
for BENCHMARK in $BENCHMARKS; do
    echo ""
    echo "--- Running $BENCHMARK ---"
    $PYTHON scripts/run_benchmark.py \
        --benchmark "$BENCHMARK" \
        --planner_engine "$MODEL_STR" \
        --planner_base_url "$PLANNER_URL" \
        --exp_name "$EXP_NAME" \
        --no_google \
        --run_all \
        --max_steps 10 \
        --max_time 300

    echo ""
    echo "--- Evaluating $BENCHMARK ---"
    $PYTHON scripts/evaluate.py \
        --benchmark "$BENCHMARK" \
        --result_dir "results/${BENCHMARK}/${EXP_NAME}" \
        --response_type direct_output
done

echo ""
echo "=== Step 3 complete for Qwen3.5-${MODEL_SIZE} ==="
