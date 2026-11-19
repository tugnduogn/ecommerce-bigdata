#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STEPS=(
  01_etl_cleaning.py
  02_prepare_gold.py
  03_rfm_kmeans.py
  04_prepare_forecast.py
  05_data_summary.py
  06_calculate_actual_revenue.py
)

echo "Đang chờ HDFS sẵn sàng..."
until docker exec namenode hdfs dfsadmin -safemode get 2>/dev/null | grep -q "OFF"; do
  sleep 5
done
echo "HDFS đã sẵn sàng."

FILTER="${1:-}"
START=$(date +%s)

for step in "${STEPS[@]}"; do
  if [[ -n "$FILTER" && "$step" != *"$FILTER"* ]]; then
    continue
  fi
  echo
  echo "=============================================="
  echo "[$(date +%T)] Chạy $step"
  echo "=============================================="
  docker exec etl python3 "/app/scripts/$step"
done

echo
echo "Pipeline hoàn tất sau $(( $(date +%s) - START )) giây."
