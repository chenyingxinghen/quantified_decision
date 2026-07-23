#!/usr/bin/env bash
# 三年训练 / 三模型 (xgb, lgb, neutral) / 两年样本内回测 / 一年样本外回测
#
# 时间窗口:
#   训练:       2022-07-21 -> 2025-07-21  (3 年)
#   样本内回测: 2023-07-21 -> 2025-07-21  (训练窗口最后 2 年, 拟合质量)
#   样本外回测: 2025-07-21 -> 2026-07-21  (训练截止后 1 年, 泛化能力)
#
# 每个模型各跑 4 组回测: 样本内(integrated + by_objective) + 样本外(integrated + by_objective)
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

TSMAX=10.0    # time-stop-max-return 设很大 -> 到点必退（分目标固定持仓）
MAXP=2        # 并发回测子进程数（按内存量力调大；同时加载多份因子缓存防 OOM）
BASE="backtest_result/exp3yr"
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"

# 模型落盘路径（训练器保存到 <SAVE_DIR>/latest/...）
XGB_PKL="$ROOT/models/exp3yr_xgb/latest/multi_objective_factor_model.pkl"
LGB_PKL="$ROOT/models/exp3yr_lgb/latest/multi_objective_factor_model.pkl"
NEU_PKL="$ROOT/models/exp3yr_neutral/latest_neural/neural_multi_objective_model.pkl"

# ───────────────────────── 1. 训练 xgb ─────────────────────────
echo "##### [$(date +%H:%M:%S)] TRAIN xgboost (GPU) #####"
GEMINI_SAVE_DIR="$ROOT/models/exp3yr_xgb" "$PY" -u scripts/train_model.py \
  --model-type xgboost --start $TRAIN_START --end $TRAIN_END \
  2>&1 | tee -a "$LOGDIR/train_xgb.log"

# ───────────────────────── 2. 训练 lgb ─────────────────────────
echo "##### [$(date +%H:%M:%S)] TRAIN lightgbm (CPU) #####"
GEMINI_SAVE_DIR="$ROOT/models/exp3yr_lgb" "$PY" -u scripts/train_model.py \
  --model-type lightgbm --start $TRAIN_START --end $TRAIN_END --skip-cache-update \
  2>&1 | tee -a "$LOGDIR/train_lgb.log"

# ──────────────────────── 3. 训练 neutral ──────────────────────
echo "##### [$(date +%H:%M:%S)] TRAIN neural (PyTorch) #####"
"$PY" -u scripts/train_neural_model.py \
  --save-dir "$ROOT/models/exp3yr_neural" --start $TRAIN_START --end $TRAIN_END --skip-cache-update \
  2>&1 | tee -a "$LOGDIR/train_neural.log"

echo "XGB_PKL=$XGB_PKL"; echo "LGB_PKL=$LGB_PKL"; echo "NEU_PKL=$NEU_PKL"

run_model_backtests () {
  local TAG="$1"; local PKL="$2"
  if [ -z "$PKL" ] || [ ! -f "$PKL" ]; then
    echo "!! [$TAG] 模型文件缺失 ($PKL)，跳过该模型全部回测"; return 1
  fi
  local PLAN
  PLAN=$("$PY" scripts/make_backtest_plan.py "$PKL")
  echo "##### [$TAG] by_objective PLAN = $PLAN"

  # 样本内 (2 年): integrated
  echo "##### [$(date +%H:%M:%S)] $TAG | 样本内 integrated"
  "$PY" -u scripts/run_backtest.py --model "$PKL" --start $IN_START --end $IN_END \
    --output "$BASE/$TAG/backtest1_in_sample/integrated" 2>&1 | tee -a "$LOGDIR/${TAG}_in_integrated.log"

  # 样本内 (2 年): by_objective
  echo "##### [$(date +%H:%M:%S)] $TAG | 样本内 by_objective"
  "$PY" -u scripts/run_backtests_parallel.py --model "$PKL" --start $IN_START --end $IN_END \
    --plan "$PLAN" --max-parallel $MAXP --time-stop-max-return $TSMAX \
    --output-base "$BASE/$TAG/backtest1_in_sample/by_objective" 2>&1 | tee -a "$LOGDIR/${TAG}_in_objective.log"

  # 样本外 (1 年): integrated
  echo "##### [$(date +%H:%M:%S)] $TAG | 样本外 integrated"
  "$PY" -u scripts/run_backtest.py --model "$PKL" --start $OUT_START --end $OUT_END \
    --output "$BASE/$TAG/backtest2_out_of_sample/integrated" 2>&1 | tee -a "$LOGDIR/${TAG}_out_integrated.log"

  # 样本外 (1 年): by_objective
  echo "##### [$(date +%H:%M:%S)] $TAG | 样本外 by_objective"
  "$PY" -u scripts/run_backtests_parallel.py --model "$PKL" --start $OUT_START --end $OUT_END \
    --plan "$PLAN" --max-parallel $MAXP --time-stop-max-return $TSMAX \
    --output-base "$BASE/$TAG/backtest2_out_of_sample/by_objective" 2>&1 | tee -a "$LOGDIR/${TAG}_out_objective.log"
}

run_model_backtests xgb    "$XGB_PKL"
run_model_backtests lgb    "$LGB_PKL"
run_model_backtests neutral "$NEU_PKL"

echo "ALL EXP3YR DONE $(date +%H:%M:%S)"
