#!/bin/bash
# ==========================================================================
# launch_experiments.sh
#
# Launch all 7 main experiments in parallel (1 GPU each), then batch eval.
#
# Usage:
#   bash launch_experiments.sh              # default: wandb online
#   bash launch_experiments.sh disabled     # wandb disabled (debug)
#
# Output structure:
#   output/{exp_name}_{TIMESTAMP}/
#     config.json
#     train_*.log
#     model-{1..10}.pt
#     sample-{1..10}.png
#     visuals/
#     eval/
#     metrics.csv
#     eval_*.log
# ==========================================================================

set -euo pipefail

WANDB_MODE="${1:-online}"
CONFIG="configs/cifar10_curriculum.yaml"
OUTPUT_ROOT="output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_MILESTONES=10

# ---- Experiment definitions: name:gpu:exp_mode ----
EXPERIMENTS=(
    "uniform:0:uniform"
    "low_to_high:1:low_to_high"
    "high_to_low:2:high_to_low"
    "mid_to_ends:3:mid_to_ends"
    "static_low_to_high:4:static_low_to_high"
    "static_high_to_low:5:static_high_to_low"
    "static_mid_to_ends:6:static_mid_to_ends"
)

echo "============================================================"
echo "  Curriculum DDPM Experiment Launch"
echo "  Time       : $(date --iso-8601=seconds)"
echo "  Timestamp  : ${TIMESTAMP}"
echo "  Config     : ${CONFIG}"
echo "  Wandb      : ${WANDB_MODE}"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "============================================================"

# ===================================================================
# Phase 1: Training (7 GPUs in parallel)
# ===================================================================
echo ""
echo "[Phase 1] Launching training..."

TRAIN_PIDS=()
NAMES=()
OUTPUT_DIRS=()
TRAIN_OK=()

for entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r name gpu exp <<< "$entry"
    output_dir="${OUTPUT_ROOT}/${name}_${TIMESTAMP}"

    echo "  GPU ${gpu} | ${name} → ${output_dir}"

    CUDA_VISIBLE_DEVICES=$gpu nohup python train_cifar10.py \
        --config "$CONFIG" \
        --exp "$exp" \
        --name "${name}_${TIMESTAMP}" \
        --output-dir "$output_dir" \
        --wandb-mode "$WANDB_MODE" \
        > /dev/null 2>&1 &

    TRAIN_PIDS+=($!)
    NAMES+=("$name")
    OUTPUT_DIRS+=("$output_dir")
done

echo ""
echo "  All launched. PIDs: ${TRAIN_PIDS[*]}"
echo "  Logs: tail -f ${OUTPUT_ROOT}/<name>_${TIMESTAMP}/train_*.log"
echo ""
echo "  Waiting for all training to complete..."

# Wait and report
FAILED=0
for i in "${!TRAIN_PIDS[@]}"; do
    if wait "${TRAIN_PIDS[$i]}"; then
        echo "  ✓ ${NAMES[$i]} (PID ${TRAIN_PIDS[$i]}) finished OK"
        TRAIN_OK[$i]=1
    else
        echo "  ✗ ${NAMES[$i]} (PID ${TRAIN_PIDS[$i]}) FAILED (exit $?)"
        FAILED=$((FAILED + 1))
        TRAIN_OK[$i]=0
    fi
done

echo ""
if [ $FAILED -gt 0 ]; then
    echo "  WARNING: ${FAILED} experiment(s) failed. Check logs."
fi
echo "[Phase 1] Training complete at $(date --iso-8601=seconds)"

# ===================================================================
# Phase 2: Evaluation (7 GPUs in parallel, each evals its own run)
# ===================================================================
echo ""
echo "[Phase 2] Launching evaluation..."

MILESTONES=$(seq 1 $NUM_MILESTONES)

EVAL_PIDS=()
EVAL_NAMES=()
for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r name gpu exp <<< "${EXPERIMENTS[$i]}"
    output_dir="${OUTPUT_DIRS[$i]}"

    # Skip eval if training failed or if the expected final checkpoint is missing.
    if [ "${TRAIN_OK[$i]}" != "1" ]; then
        echo "  SKIP ${name} (training failed)"
        continue
    fi

    if [ ! -f "${output_dir}/model-${NUM_MILESTONES}.pt" ]; then
        echo "  SKIP ${name} (missing ${output_dir}/model-${NUM_MILESTONES}.pt)"
        continue
    fi

    echo "  GPU ${gpu} | eval ${name} (milestones 1-${NUM_MILESTONES})"

    CUDA_VISIBLE_DEVICES=$gpu nohup python eval_cifar10.py \
        --run-dir "$output_dir" \
        --milestones $MILESTONES \
        --num-fid-samples 50000 \
        --batch-size 256 \
        > /dev/null 2>&1 &

    EVAL_PIDS+=($!)
    EVAL_NAMES+=("$name")
done

echo ""
echo "  Eval PIDs: ${EVAL_PIDS[*]}"
echo "  Waiting for all eval to complete..."

EVAL_FAILED=0
for i in "${!EVAL_PIDS[@]}"; do
    if wait "${EVAL_PIDS[$i]}"; then
        echo "  ✓ eval ${EVAL_NAMES[$i]} (PID ${EVAL_PIDS[$i]}) finished OK"
    else
        echo "  ✗ eval ${EVAL_NAMES[$i]} (PID ${EVAL_PIDS[$i]}) FAILED (exit $?)"
        EVAL_FAILED=$((EVAL_FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "  All done at $(date --iso-8601=seconds)"
echo "  Results: ${OUTPUT_ROOT}/*_${TIMESTAMP}/metrics.csv"
if [ $FAILED -gt 0 ] || [ $EVAL_FAILED -gt 0 ]; then
    echo "  WARNING: train failures=${FAILED}, eval failures=${EVAL_FAILED}"
fi
echo "============================================================"

if [ $FAILED -gt 0 ] || [ $EVAL_FAILED -gt 0 ]; then
    exit 1
fi
