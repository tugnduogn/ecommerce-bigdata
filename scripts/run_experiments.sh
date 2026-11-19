#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

MLLIB=(
  01_linear_regression.py
  02_decision_tree.py
  02_decision_tree_tuning.py
  02_decision_tree_tuning_v1.py
  02_decision_tree_tuning_v2.py
  03_random_forest.py
  03_random_forest_tuning_v1.py
)
SKLEARN=(
  prepare_forecast.py
  01_linear_regression.py
  02_decision_tree.py
  02_decision_tree_tuning_v1.py
  03_random_forest.py
  03_random_forest_tuning_v1.py
  04_extra_trees.py
)
TARGET="${1:-all}"
LOG_DIR="logs/experiments"
mkdir -p "$LOG_DIR"

RESULTS=()
FAILED=0

run_one() {
  local label="$1"; shift
  local log="$LOG_DIR/${label//\//_}.log"
  echo
  echo "=============================================="
  echo "[$(date +%T)] $label"
  echo "=============================================="
  if "$@" 2>&1 | tee "$log"; then
    RESULTS+=("OK     $label")
  else
    RESULTS+=("LOI    $label  (xem $log)")
    FAILED=$((FAILED + 1))
  fi
}

START=$(date +%s)

if [[ "$TARGET" == "all" || "$TARGET" == "mllib" ]]; then
  for f in "${MLLIB[@]}"; do
    run_one "mllib/$f" docker exec etl python3 "/app/scripts/experiments/mllib/$f"
  done
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sklearn" ]]; then
  for f in "${SKLEARN[@]}"; do
    run_one "sklearn/$f" docker exec ml python3 "/app/$f"
  done
fi

echo
echo "================ TỔNG KẾT ================"
for r in "${RESULTS[@]}"; do echo "$r"; done
echo "Thời gian: $(( $(date +%s) - START )) giây"

[[ "$FAILED" -eq 0 ]]
