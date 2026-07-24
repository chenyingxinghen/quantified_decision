#!/usr/bin/env bash
# exp3yr 分析阶段：对每个已训练的模型 (xgb / lgb / neutral) 跑 analysis/run_analysis.py
# 依赖 run_exp3yr_resilient.sh 先训出 models/exp3yr_xgb|lgb|neutral。
# 跨平台：自动探测 python；分析默认 3000 只（64GB cgroup 安全），可用 STOCKS 覆盖。
# 数据集只构建一次（首个模型建并缓存，其余复用 --dataset）。
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Windows(git bash) 下 pwd 给出 /g/... 的 posix 路径，原生 Windows Python 无法解析
# （会当成 G:\g\... 找不到）。转成 G:/... 形式，保证 --model/--dataset 等路径可被正确打开。
if [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W)"
fi
cd "$SCRIPT_DIR"

if [[ "${OS:-}" == "Windows_NT" ]] || [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
  PY="G:/quantified_decision/.venv/Scripts/python.exe"
else
  PY="${PYTHON:-python}"
fi

ANALYSIS_START=2022-07-21   # 覆盖 训练3年 + 样本内2年 + 样本外1年
ANALYSIS_END=2026-07-21
BASE="backtest_result/exp3yr"
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR" "$BASE/analysis"
DATASET_CACHE="$SCRIPT_DIR/analysis/full_dataset_exp3yr"   # 目录式缓存（多文件），xgb 建一次、lgb/neutral 复用
STOCKS="${STOCKS:-3000}"   # 64GB cgroup 安全；如需全量 STOCKS=6000（分析可能 OOM）

XGB_PKL="$SCRIPT_DIR/models/exp3yr_xgb/latest/multi_objective_factor_model.pkl"
LGB_PKL="$SCRIPT_DIR/models/exp3yr_lgb/latest/multi_objective_factor_model.pkl"
NEU_PKL="$SCRIPT_DIR/models/exp3yr_neutral/latest_neural/neural_multi_objective_model.pkl"

# ── 等待主流程全部回测结束（neutral 样本外 by_objective 产出 metrics 即视为完成）──
# SKIP_WAIT=1 可跳过等待（已确认相关模型回测完成 / 不想等 neutral 时）
SENTINEL_DIR="$SCRIPT_DIR/backtest_result/exp3yr/neutral/backtest2_out_of_sample/by_objective"
MAX_WAIT=28800   # 最多等 8 小时
if [ "${SKIP_WAIT:-0}" != "1" ]; then
  ELAPSED=0
  echo "[analysis-wait] 等待主流程回测结束 (sentinel: $SENTINEL_DIR/*/backtest_metrics.json)"
  while :; do
    if ls "$SENTINEL_DIR"/*/backtest_metrics.json >/dev/null 2>&1; then
      echo "[analysis-wait] 检测到 sentinel，开始分析"; break
    fi
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
      echo "[analysis-wait] 超时($MAX_WAIT s)，用已有模型继续执行分析"; break
    fi
    sleep 120
    ELAPSED=$((ELAPSED + 120))
  done
else
  echo "[analysis-wait] SKIP_WAIT=1，跳过等待直接开始分析"
fi

# RUN_ONLY 过滤：逗号分隔，如 RUN_ONLY=xgb,lgb；为空则全跑
RUN_ONLY="${RUN_ONLY:-}"
should_run () {
  local TAG="$1"
  [ -z "$RUN_ONLY" ] && return 0
  printf '%s' "$RUN_ONLY" | tr ',' '\n' | grep -qx "$TAG"
}

run_one_analysis () {
  local TAG="$1"; local PKL="$2"; local EXTRA="$3"
  if [ ! -f "$PKL" ]; then
    echo "!! [$TAG] 模型文件缺失 ($PKL)，跳过分析"; return 1
  fi
  local OUT="$BASE/analysis/$TAG"
  mkdir -p "$OUT"
  echo "##### [$(date +%H:%M:%S)] ANALYSE $TAG -> $OUT"
  PYTHONPATH="$SCRIPT_DIR" "$PY" -u -m analysis.run_analysis \
    --model "$PKL" --out "$OUT" \
    --stocks "$STOCKS" \
    --start $ANALYSIS_START --end $ANALYSIS_END \
    --dataset "$DATASET_CACHE" --cache "$DATASET_CACHE" \
    $EXTRA 2>&1 | tee -a "$LOGDIR/analysis_${TAG}.log"
}

# 数据集只构建一次（首个模型构建并缓存，其余复用 --dataset）
should_run xgb    && run_one_analysis xgb    "$XGB_PKL" ""
should_run lgb    && run_one_analysis lgb    "$LGB_PKL" ""
should_run neutral && run_one_analysis neutral "$NEU_PKL" "--no-shap"

echo "ALL EXP3YR ANALYSIS DONE $(date +%H:%M:%S)"
