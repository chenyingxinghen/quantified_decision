#!/usr/bin/env bash
# 断点续跑版：每个阶段检查产物是否已存在，存在则跳过。
# 用法: 进程挂掉后重跑本脚本即可从断点继续。
# 单目标(by_objective)回测只测 20d 收益 (rank_y_ret_20d)，其余目标跳过；
# 已回测完成的模型 (.backtests_done 存在) 整体跳过。
# 跨平台：自动探测 python（Windows 用 venv，云端用环境 python）。
set -u
# 日志编码统一为 UTF-8
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${OS:-}" == "Windows_NT" ]] || [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
  PY="G:/quantified_decision/.venv/Scripts/python.exe"
else
  PY="${PYTHON:-python}"
fi
ROOT="$SCRIPT_DIR"

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
    --model-type xgboost --stocks 6000 --start $TRAIN_START --end $TRAIN_END \
    2>&1 | tee -a "$LOGDIR/train_xgb.log"
}
stage_lgb () {
  if [ -f "$LGB_PKL" ]; then echo "[$(date +%H:%M:%S)] [skip] lgb 模型已存在，跳过"; return 0; fi
  echo "##### [$(date +%H:%M:%S)] TRAIN lightgbm (CPU) #####"
  GEMINI_SAVE_DIR="$ROOT/models/exp3yr_lgb" "$PY" -u scripts/train_model.py \
    --model-type lightgbm --stocks 6000 --start $TRAIN_START --end $TRAIN_END --skip-cache-update \
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
  echo "##### [$TAG] full PLAN = $PLAN"

  local PLAN_20D
  PLAN_20D=$(echo "$PLAN" | tr ',' '\n' | grep -E 'ret_20d:[0-9]+$' | paste -sd, -)
  if [ -z "$PLAN_20D" ]; then
    echo "!! [$TAG] 未找到 rank_y_ret_20d 目标，跳过全部单目标(by_objective)回测"
  else
    echo "##### [$TAG] by_objective (仅 20d 收益) PLAN = $PLAN_20D"
  fi

  echo "##### [$(date +%H:%M:%S)] $TAG | 样本内 integrated"
  "$PY" -u scripts/run_backtest.py --model "$PKL" --start $IN_START --end $IN_END \
    --output "$BASE/$TAG/backtest1_in_sample/integrated" 2>&1 | tee -a "$LOGDIR/${TAG}_in_integrated.log"

  if [ -n "$PLAN_20D" ]; then
    echo "##### [$(date +%H:%M:%S)] $TAG | 样本内 by_objective (仅 20d 收益)"
    "$PY" -u scripts/run_backtests_parallel.py --model "$PKL" --start $IN_START --end $IN_END \
      --plan "$PLAN_20D" --max-parallel $MAXP --time-stop-max-return $TSMAX \
      --output-base "$BASE/$TAG/backtest1_in_sample/by_objective" 2>&1 | tee -a "$LOGDIR/${TAG}_in_objective.log"
  fi

  echo "##### [$(date +%H:%M:%S)] $TAG | 样本外 integrated"
  "$PY" -u scripts/run_backtest.py --model "$PKL" --start $OUT_START --end $OUT_END \
    --output "$BASE/$TAG/backtest2_out_of_sample/integrated" 2>&1 | tee -a "$LOGDIR/${TAG}_out_integrated.log"

  if [ -n "$PLAN_20D" ]; then
    echo "##### [$(date +%H:%M:%S)] $TAG | 样本外 by_objective (仅 20d 收益)"
    "$PY" -u scripts/run_backtests_parallel.py --model "$PKL" --start $OUT_START --end $OUT_END \
      --plan "$PLAN_20D" --max-parallel $MAXP --time-stop-max-return $TSMAX \
      --output-base "$BASE/$TAG/backtest2_out_of_sample/by_objective" 2>&1 | tee -a "$LOGDIR/${TAG}_out_objective.log"
  fi

  touch "$SENT"
  echo "##### [$(date +%H:%M:%S)] $TAG 回测完成"
}

# 可选：仅运行指定模型（逗号分隔，如 RUN_ONLY=lgb）
RUN_ONLY="${RUN_ONLY:-}"
should_run () {
  [ -z "$RUN_ONLY" ] && return 0
  printf '%s' "$RUN_ONLY" | tr ',' '\n' | grep -qx "$1"
}

should_run xgb    && stage_xgb
should_run lgb    && stage_lgb
should_run neutral && stage_neutral
should_run xgb    && run_model_backtests xgb    "$XGB_PKL"
should_run lgb    && run_model_backtests lgb    "$LGB_PKL"
should_run neutral && run_model_backtests neutral "$NEU_PKL"

echo "ALL EXP3YR DONE $(date +%H:%M:%S)"
