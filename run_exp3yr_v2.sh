#!/usr/bin/env bash
# ============================================================================
# run_exp3yr_v2.sh — 一键产出 exp3yr_v2 全量产物
#   顺序：训练(xgb/lgb/neutral) → 回测(样本内+样本外, integrated+by_objective) → 分析(report)
#   合并自 run_exp3yr_resilient.sh(训练+回测) 与 run_exp3yr_analysis.sh(分析)。
#
# 关键：本脚本导出 QD_MODEL_VERSION=v2，启用 v2 正则化加强超参与剔除退化目标
#       (illiq/tradable) 的权重，直接针对上一版 OOS 全面亏损的回测结论。
#
# 云端用法（推荐）：
#   cd /gemini/code/quantified_decision
#   export GEMINI_DATA_IN1=/gemini/data-1      # 真实库所在（数据已在 data-1）
#   unset GEMINI_DATA_OUT                      # DB 路径回退到 data-1，避免空壳库
#   bash run_exp3yr_v2.sh                       # 全跑 xgb+lgb+neutral
#   # 只跑 xgb/lgb（neutral 需 GPU 且慢）：
#   RUN_ONLY=xgb,lgb bash run_exp3yr_v2.sh
#
# 断点续跑 / 跳过：
#   SKIP_TRAIN=1       只跑回测+分析（模型已训好）
#   SKIP_BACKTEST=1    只跑训练+分析
#   SKIP_ANALYSIS=1    只跑训练+回测
#   RUN_ONLY=xgb,lgb,neutral   仅指定模型（逗号分隔）
#   STOCKS=6000                  分析股票数（默认 3000，64GB cgroup 安全；6000 可能 OOM）
#   任何已完成阶段（模型 pkl 存在 / .backtests_done / report.md 存在）会自动跳过。
#
# 产物：
#   模型   : models/exp3yr_v2_{xgb,lgb,neutral}/latest/...
#   回测   : backtest_result/exp3yr_v2/{xgb,lgb,neutral}/...
#   分析报告: backtest_result/exp3yr_v2/analysis/{xgb,lgb,neutral}/report.md + metrics.csv + figures/
# ============================================================================
set -u
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export QD_MODEL_VERSION=v2            # 关键：启用 v2 训练配置

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Windows(git bash) 下 pwd 给出 /g/... 的 posix 路径，原生 Windows Python 无法解析，
# 转成 G:/... 形式，保证 --model/--dataset 等路径被正确打开。
if [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W)"
fi
cd "$SCRIPT_DIR"

if [[ "${OS:-}" == "Windows_NT" ]] || [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
  PY="G:/quantified_decision/.venv/Scripts/python.exe"
else
  PY="${PYTHON:-python}"
fi

# ── 时间窗口 ───────────────────────────────────────────────────────────────
TRAIN_START=2022-07-21
TRAIN_END=2025-07-21
IN_START=2023-07-21
IN_END=2025-07-21
OUT_START=2025-07-21
OUT_END=2026-07-21

TSMAX=10.0
MAXP=2
BASE="backtest_result/exp3yr_v2"
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"

# 分析股票数（64GB cgroup 安全）；如需全量 STOCKS=6000（分析可能 OOM）
ASTOCKS="${STOCKS:-3000}"
ANALYSIS_START=2022-07-21
ANALYSIS_END=2026-07-21
DATASET_CACHE="$SCRIPT_DIR/analysis/full_dataset_exp3yr_v2"   # 目录式缓存：xgb 建一次、lgb/neutral 复用

XGB_PKL="$SCRIPT_DIR/models/exp3yr_v2_xgb/latest/multi_objective_factor_model.pkl"
LGB_PKL="$SCRIPT_DIR/models/exp3yr_v2_lgb/latest/multi_objective_factor_model.pkl"
NEU_PKL="$SCRIPT_DIR/models/exp3yr_v2_neutral/latest_neural/neural_multi_objective_model.pkl"

# ── RUN_ONLY 过滤 ──────────────────────────────────────────────────────────
RUN_ONLY="${RUN_ONLY:-}"
should_run () {
  [ -z "$RUN_ONLY" ] && return 0
  printf '%s' "$RUN_ONLY" | tr ',' '\n' | grep -qx "$1"
}

# ── 训练阶段（带断点续跑：pkl 已存在则跳过；GBM 带 --resume）────────────────
# 重要：lgb 在 64GB cgroup 下用 6000 只容易 OOM（joblib 6+ 进程 × path_smooth=10 × 5 目标），
# 默认 3000 只；如机器更大可 LGB_STOCKS=6000 bash run_exp3yr_v2.sh。
LGB_STOCKS="${LGB_STOCKS:-3000}"
stage_xgb () {
  if [ "${SKIP_TRAIN:-0}" = "1" ]; then echo "[$(date +%H:%M:%S)] [skip] SKIP_TRAIN"; return 0; fi
  if [ -f "$XGB_PKL" ]; then echo "[$(date +%H:%M:%S)] [skip] xgb 模型已存在，跳过"; return 0; fi
  echo "##### [$(date +%H:%M:%S)] TRAIN xgboost (GPU, v2) #####"
  GEMINI_SAVE_DIR="$SCRIPT_DIR/models/exp3yr_v2_xgb" "$PY" -u scripts/train_model.py \
    --model-type xgboost --stocks 6000 --start $TRAIN_START --end $TRAIN_END --resume \
    2>&1 | tee -a "$LOGDIR/train_xgb.log"
}
stage_lgb () {
  if [ "${SKIP_TRAIN:-0}" = "1" ]; then echo "[$(date +%H:%M:%S)] [skip] SKIP_TRAIN"; return 0; fi
  if [ -f "$LGB_PKL" ]; then echo "[$(date +%H:%M:%S)] [skip] lgb 模型已存在，跳过"; return 0; fi
  echo "##### [$(date +%H:%M:%S)] TRAIN lightgbm (CPU, v2, stocks=$LGB_STOCKS) #####"
  GEMINI_SAVE_DIR="$SCRIPT_DIR/models/exp3yr_v2_lgb" "$PY" -u scripts/train_model.py \
    --model-type lightgbm --stocks "$LGB_STOCKS" --start $TRAIN_START --end $TRAIN_END \
    --skip-cache-update --resume \
    2>&1 | tee -a "$LOGDIR/train_lgb.log"
  # lgb 训练完成度自检：日志最末若没有 "训练完成" 字样则标记失败，便于上层重试
  if ! tail -50 "$LOGDIR/train_lgb.log" 2>/dev/null | grep -qE "训练完成|=== 多目标模型训练完成"; then
    echo "!! [$(date +%H:%M:%S)] lgb 训练未完成（检查 $LOGDIR/train_lgb.log），建议降低 LGB_STOCKS 重跑"
  fi
}
stage_neutral () {
  if [ "${SKIP_TRAIN:-0}" = "1" ]; then echo "[$(date +%H:%M:%S)] [skip] SKIP_TRAIN"; return 0; fi
  if [ -f "$NEU_PKL" ]; then echo "[$(date +%H:%M:%S)] [skip] neutral 模型已存在，跳过"; return 0; fi
  echo "##### [$(date +%H:%M:%S)] TRAIN neural (v2) #####"
  "$PY" -u scripts/train_neural_model.py \
    --save-dir "$SCRIPT_DIR/models/exp3yr_v2_neutral" --start $TRAIN_START --end $TRAIN_END --skip-cache-update \
    2>&1 | tee -a "$LOGDIR/train_neural.log"
}

# ── 回测阶段 ───────────────────────────────────────────────────────────────
run_model_backtests () {
  local TAG="$1"; local PKL="$2"
  if [ "${SKIP_BACKTEST:-0}" = "1" ]; then echo "[$(date +%H:%M:%S)] [skip] SKIP_BACKTEST"; return 0; fi
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
    echo "!! [$TAG] 未找到 rank_y_ret_20d 目标，跳过单目标(by_objective)回测"
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

# ── 分析阶段（report.md 已存在则跳过）──────────────────────────────────────
run_one_analysis () {
  local TAG="$1"; local PKL="$2"; local EXTRA="$3"
  if [ "${SKIP_ANALYSIS:-0}" = "1" ]; then echo "[$(date +%H:%M:%S)] [skip] SKIP_ANALYSIS"; return 0; fi
  if [ ! -f "$PKL" ]; then echo "!! [$TAG] 模型文件缺失 ($PKL)，跳过分析"; return 1; fi
  local OUT="$BASE/analysis/$TAG"
  if [ -f "$OUT/report.md" ]; then echo "[$(date +%H:%M:%S)] [skip] $TAG 报告已存在"; return 0; fi
  mkdir -p "$OUT"
  echo "##### [$(date +%H:%M:%S)] ANALYSE $TAG -> $OUT"
  PYTHONPATH="$SCRIPT_DIR" "$PY" -u -m analysis.run_analysis \
    --model "$PKL" --out "$OUT" \
    --stocks "$ASTOCKS" \
    --start $ANALYSIS_START --end $ANALYSIS_END \
    --dataset "$DATASET_CACHE" --cache "$DATASET_CACHE" \
    $EXTRA 2>&1 | tee -a "$LOGDIR/analysis_${TAG}.log"
}

# ── 编排 ───────────────────────────────────────────────────────────────────
echo "========== EXP3YR V2 启动 (QD_MODEL_VERSION=$QD_MODEL_VERSION) =========="
should_run xgb     && stage_xgb
should_run lgb     && stage_lgb
should_run neutral && stage_neutral
should_run xgb     && run_model_backtests xgb     "$XGB_PKL"
should_run lgb     && run_model_backtests lgb     "$LGB_PKL"
should_run neutral && run_model_backtests neutral "$NEU_PKL"
should_run xgb     && run_one_analysis xgb     "$XGB_PKL" ""
should_run lgb     && run_one_analysis lgb     "$LGB_PKL" ""
should_run neutral && run_one_analysis neutral "$NEU_PKL" "--no-shap"

echo "ALL EXP3YR V2 DONE $(date +%H:%M:%S)"
