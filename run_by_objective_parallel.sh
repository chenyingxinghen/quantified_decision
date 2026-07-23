#!/usr/bin/env bash
# 按训练目标分别回测（固定持仓天数 = 各目标 label horizon），并行版
# 使用 scripts/run_backtests_parallel.py 并发启动多个回测子进程
set -u
cd "G:/quantified_decision"
PY="G:/quantified_decision/.venv/Scripts/python.exe"
MODEL="models/latest/multi_objective_factor_model.pkl"
START=2026-01-23
END=2026-07-22
TSMAX=10.0   # time-stop-max-return 设很大 -> 到点必退（固定持仓）
MAXP=2       # 并发回测子进程数（按内存量力调大，避免同时加载多份因子缓存 OOM）
LOG="backtest_by_objective_parallel.log"

PLAN="rank_y_ret_5d:5,rank_y_ret_20d:20,rank_y_ret_60d:60,rank_y_mdd_20d:20,rank_y_downvol_20d:20,rank_y_illiq_20d:20"

"$PY" -u scripts/run_backtests_parallel.py \
  --model "$MODEL" --start "$START" --end "$END" \
  --plan "$PLAN" --max-parallel "$MAXP" --time-stop-max-return "$TSMAX" \
  --output-base "backtest_result/by_objective" 2>&1 | tee "$LOG"

echo "ALL OBJECTIVES DONE $(date +%H:%M:%S)" | tee -a "$LOG"
