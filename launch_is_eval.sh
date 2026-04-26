#!/usr/bin/env bash
set -u

TIMESTAMP="${1:-20260426_030255}"
NUM_IS_SAMPLES="${NUM_IS_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MILESTONES=(1 2 3 4 5 6 7 8 9 10)

EXPERIMENTS=(
  uniform
  low_to_high
  high_to_low
  mid_to_ends
  static_low_to_high
  static_high_to_low
  static_mid_to_ends
)

echo "============================================================"
echo "  Inception Score Eval Launch"
echo "  Time          : $(date --iso-8601=seconds)"
echo "  Timestamp     : ${TIMESTAMP}"
echo "  Num IS samples: ${NUM_IS_SAMPLES}"
echo "  Batch size    : ${BATCH_SIZE}"
echo "============================================================"

PIDS=()
NAMES=()
FAILURES=0

for i in "${!EXPERIMENTS[@]}"; do
  EXP="${EXPERIMENTS[$i]}"
  RUN_DIR="output/${EXP}_${TIMESTAMP}"

  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "  ✗ missing run dir: ${RUN_DIR}"
    FAILURES=$((FAILURES + 1))
    continue
  fi

  if [[ -e "${RUN_DIR}/metrics_is.csv" ]]; then
    echo "  ✗ refusing to append existing file: ${RUN_DIR}/metrics_is.csv"
    FAILURES=$((FAILURES + 1))
    continue
  fi

  echo "  GPU ${i} | IS eval ${EXP} → ${RUN_DIR}/metrics_is.csv"
  CUDA_VISIBLE_DEVICES="${i}" python eval_cifar10.py \
    --run-dir "${RUN_DIR}" \
    --milestones "${MILESTONES[@]}" \
    --compute-is \
    --skip-fid \
    --skip-visuals \
    --metrics-file metrics_is.csv \
    --num-is-samples "${NUM_IS_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" &

  PIDS+=("$!")
  NAMES+=("${EXP}")
done

echo
echo "  IS eval PIDs: ${PIDS[*]}"
echo "  Waiting for all IS eval jobs to complete..."

for i in "${!PIDS[@]}"; do
  PID="${PIDS[$i]}"
  NAME="${NAMES[$i]}"
  if wait "${PID}"; then
    echo "  ✓ IS eval ${NAME} (PID ${PID}) finished OK"
  else
    echo "  ✗ IS eval ${NAME} (PID ${PID}) failed"
    FAILURES=$((FAILURES + 1))
  fi
done

echo
if [[ "${FAILURES}" -eq 0 ]]; then
  echo "All IS eval jobs finished successfully."
else
  echo "${FAILURES} IS eval job(s) failed or had missing run dirs."
  exit 1
fi
