#!/usr/bin/env bash
# Step 3: Run all 5 Qwen3.5 models on all 5 benchmarks.
#
# Usage:
#   bash scripts/run_step3.sh <model_size> <planner_base_url>
#
# Example:
#   bash scripts/run_step3.sh 0.8b "https://varunkasa77--agentflow-planner-qwen35-0.8b-serve.modal.run/v1"
#   bash scripts/run_step3.sh 2b   "https://varunkasa77--agentflow-planner-qwen35-2b-serve.modal.run/v1"
#   bash scripts/run_step3.sh 4b   "https://varunkasa77--agentflow-planner-qwen35-4b-serve.modal.run/v1"
#   bash scripts/run_step3.sh 9b   "https://varunkasa77--agentflow-planner-qwen35-9b-serve.modal.run/v1"
#   bash scripts/run_step3.sh 27b  "https://varunkasa77--agentflow-planner-a100-serve.modal.run/v1"

MODEL_SIZE=${1:-"0.8b"}
PLANNER_URL=${2}
PYTHON=".venv/Scripts/python.exe"
EXP_NAME="step3_qwen35_${MODEL_SIZE}"
MODEL_STR="vllm-Qwen/Qwen3.5-${MODEL_SIZE^^}-Instruct"

if [ -z "$PLANNER_URL" ]; then
    echo "Usage: bash scripts/run_step3.sh <model_size> <planner_base_url>"
    exit 1
fi

echo "=== Step 3: Qwen3.5-${MODEL_SIZE} ==="
echo "Planner URL: $PLANNER_URL"
echo "Exp name: $EXP_NAME"

for BENCHMARK in bamboogle twowiki hotpotqa musique gaia; do
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
