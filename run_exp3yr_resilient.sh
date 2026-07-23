#!/usr/bin/env bash
# 断点续跑版：每个阶段检查产物是否已存在，存在则跳过。
# 用法: 进程挂掉后重跑本脚本即可从断点继续。
set -u
cd "G:/quantified_decision"
PY="G:/quantified_decision/.venv/Scripts/python.exe"
ROOT="G:/quantified_decision"

TRAIN_START=2022-07-21
TRAIN_END=2025-07-21
IN_START=2023-07-21
IN_END=2025-07-21
OUT_START=2025-07-21
OUT_END=2026-07-21

TSMAX=10.0
MAXP=2
BASE="backtest_result/exp3yr"
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"

XGB_PKL="$ROOT/models/exp3yr_xgb/latest/multi_objective_factor_model.pkl"
LGB_PKL="$ROOT/models/exp3yr_lgb/latest/multi_objective_factor_model.pkl"
NEU_PKL="$ROOT/models/exp3yr_neutral/latest_neural/neural_multi_objective_model.pkl"

stage_xgb () {
  if [ -f "$XGB_PKL" ]; then echo "[$(date +%H:%M:%S)] [skip] xgb 模型已存在，跳过"; return 0; fi
  echo "##### [$(date +%H:%M:%S)] TRAIN xgboost (GPU) #####"
  GEMINI_SAVE_DIR="$ROOT/models/exp3yr_xgb" "$PY" -u scripts/train_model.py \
    --model-type xgboost --start $TRAIN_START --end $TRAIN_END \
    2>&1 | tee -a "$LOGDIR/train_xgb.log"
}

stage_lgb () {
  if [ -f "$LGB_PKL" ]; then echo "[$(date +%H:%M:%S)] [skip] lgb 模型已存在，跳过"; return 0; fi
  echo "##### [$(date +%H:%M:%S)] TRAIN lightgbm (CPU) #####"
  GEMINI_SAVE_DIR="$ROOT/models/exp3yr_lgb" "$PY" -u scripts/train_model.py \
    --model-type lightgbm --start $TRAIN_START --end $TRAIN_END --skip-cache-update \
    2>&1 | tee -a "$LOGDIR/train_lgb.log"
}

stage_neutral () {
  if [ -f "$NEU_PKL" ]; then echo "[$(date +%H:%M:%S)] [skip] neutral 模型已存在，跳过"; return 0; fi
  echo "##### [$(date +%H:%M:%S)] TRAIN neural (PyTorch) #####"
  "$PY" -u scripts/train_neural_model.py \
    --save-dir "$ROOT/models/exp3yr_neutral" --start $TRAIN_START --end $TRAIN_END --skip-cache-update \
    2>&1 | tee -a "$LOGDIR/train_neural.log"
}

run_model_backtests () {
  local TAG="$1"; local PKL="$2"
  if [ -z "$PKL" ] || [ ! -f "$PKL" ]; then
    echo "!! [$TAG] 模型文件缺失 ($PKL)，跳过该模型全部回测"; return 1
  fi
  local SENT="$BASE/$TAG/.backtests_done"
  if [ -f "$SENT" ]; then echo "[$(date +%H:%M:%S)] [skip] $TAG 回测已完成，跳过"; return 0; fi

  local PLAN
  PLAN=$("$PY" scripts/make_backtest_plan.py "$PKL")
  echo "##### [$TAG] by_objective PLAN = $PLAN"

  echo "##### [$(date +%H:%M:%S)] $TAG | 样本内 integrated"
  "$PY" -u scripts/run_backtest.py --model "$PKL" --start $IN_START --end $IN_END \
    --output "$BASE/$TAG/backtest1_in_sample/integrated" 2>&1 | tee -a "$LOGDIR/${TAG}_in_integrated.log"

  echo "##### [$(date +%H:%M:%S)] $TAG | 样本内 by_objective"
  "$PY" -u scripts/run_backtests_parallel.py --model "$PKL" --start $IN_START --end $IN_END \
    --plan "$PLAN" --max-parallel $MAXP --time-stop-max-return $TSMAX \
    --output-base "$BASE/$TAG/backtest1_in_sample/by_objective" 2>&1 | tee -a "$LOGDIR/${TAG}_in_objective.log"

  echo "##### [$(date +%H:%M:%S)] $TAG | 样本外 integrated"
  "$PY" -u scripts/run_backtest.py --model "$PKL" --start $OUT_START --end $OUT_END \
    --output "$BASE/$TAG/backtest2_out_of_sample/integrated" 2>&1 | tee -a "$LOGDIR/${TAG}_out_integrated.log"

  echo "##### [$(date +%H:%M:%S)] $TAG | 样本外 by_objective"
  "$PY" -u scripts/run_backtests_parallel.py --model "$PKL" --start $OUT_START --end $OUT_END \
    --plan "$PLAN" --max-parallel $MAXP --time-stop-max-return $TSMAX \
    --output-base "$BASE/$TAG/backtest2_out_of_sample/by_objective" 2>&1 | tee -a "$LOGDIR/${TAG}_out_objective.log"

  touch "$SENT"
  echo "##### [$(date +%H:%M:%S)] $TAG 回测完成"
}

stage_xgb
stage_lgb
stage_neutral
run_model_backtests xgb    "$XGB_PKL"
run_model_backtests lgb    "$LGB_PKL"
run_model_backtests neutral "$NEU_PKL"

echo "ALL EXP3YR DONE $(date +%H:%M:%S)"
