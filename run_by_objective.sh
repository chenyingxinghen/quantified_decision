#!/usr/bin/env bash
# 按训练目标分别回测（固定持仓天数 = 各目标 label horizon）
set -u
cd "G:/quantified_decision"
PY="G:/quantified_decision/.venv/Scripts/python.exe"
MODEL="models/latest/multi_objective_factor_model.pkl"
START=2026-01-23
END=2026-07-22
TSMAX=10.0   # time-stop-max-return 设很大 -> 到点必退（固定持仓）
LOG="backtest_by_objective.log"

run_one() {
  local obj="$1" days="$2"
  echo "===== START $obj hold=$days $(date +%H:%M:%S) =====" | tee -a "$LOG"
  "$PY" -u scripts/run_backtest.py --model "$MODEL" --start "$START" --end "$END" \
    --objective "$obj" --hold-days "$days" --time-stop-max-return "$TSMAX" \
    --output "backtest_result/by_objective/${obj}_hold${days}" >> "$LOG" 2>&1
  echo "===== DONE $obj exit=$? $(date +%H:%M:%S) =====" | tee -a "$LOG"
}

run_one rank_y_ret_5d 5
run_one rank_y_ret_20d 20
run_one rank_y_ret_60d 60
run_one rank_y_mdd_20d 20
run_one rank_y_downvol_20d 20
run_one rank_y_illiq_20d 20

echo "ALL OBJECTIVES DONE $(date +%H:%M:%S)" | tee -a "$LOG"
