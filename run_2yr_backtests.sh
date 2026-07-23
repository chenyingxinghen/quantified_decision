#!/usr/bin/env bash
# 2年训练实验的回测编排：回测1(同2年,样本内) + 回测2(其后1年,前向样本外)
# 每个回测均含「集成预测」(combined) 与「6个分目标预测」(per-objective, 固定持仓=标签周期)
# 依赖：先跑完 run_2yr_training (模型落到 models/latest/multi_objective_factor_model.pkl)
set -u
cd "G:/quantified_decision"
PY="G:/quantified_decision/.venv/Scripts/python.exe"
MODEL="models/latest/multi_objective_factor_model.pkl"
TSMAX=10.0   # time-stop-max-return 设很大 -> 到点必退（分目标固定持仓）
MAXP=3       # 并发回测子进程数（按内存量力调大）
PLAN="rank_y_ret_5d:5,rank_y_ret_20d:20,rank_y_ret_60d:60,rank_y_mdd_20d:20,rank_y_downvol_20d:20,rank_y_illiq_20d:20"
BASE="backtest_result/exp2yr"

if [ ! -f "$MODEL" ]; then
  echo "错误: 模型不存在 $MODEL，请先运行训练"; exit 2
fi

echo "===== 回测1 (同2年, 样本内): 2023-07-21 -> 2025-07-21 ====="
"$PY" -u scripts/run_backtest.py --model "$MODEL" --start 2023-07-21 --end 2025-07-21 \
  --output "$BASE/backtest1/integrated" 2>&1 | tee -a "$BASE/backtest1_integrated.log"
"$PY" -u scripts/run_backtests_parallel.py --model "$MODEL" --start 2023-07-21 --end 2025-07-21 \
  --plan "$PLAN" --max-parallel "$MAXP" --time-stop-max-return "$TSMAX" \
  --output-base "$BASE/backtest1/by_objective" 2>&1 | tee -a "$BASE/backtest1_objective.log"

echo "===== 回测2 (其后1年, 前向样本外): 2025-07-21 -> 2026-07-21 ====="
"$PY" -u scripts/run_backtest.py --model "$MODEL" --start 2025-07-21 --end 2026-07-21 \
  --output "$BASE/backtest2/integrated" 2>&1 | tee -a "$BASE/backtest2_integrated.log"
"$PY" -u scripts/run_backtests_parallel.py --model "$MODEL" --start 2025-07-21 --end 2026-07-21 \
  --plan "$PLAN" --max-parallel "$MAXP" --time-stop-max-return "$TSMAX" \
  --output-base "$BASE/backtest2/by_objective" 2>&1 | tee -a "$BASE/backtest2_objective.log"

echo "ALL BACKTESTS DONE $(date +%H:%M:%S)"
