#!/usr/bin/env bash
# exp3yr 分析阶段：对每个模型 (xgb / lgb / neutral) 跑 analysis/run_analysis.py
# 仅依赖已训练好的模型 pkl，与回测无输出依赖；本脚本等待主流程
# (run_exp3yr.sh) 全部回测结束后再启动，避免与回测争夺 DB / 因子缓存 / 内存。
set -u
cd "G:/quantified_decision"
PY="G:/quantified_decision/.venv/Scripts/python.exe"
ROOT="G:/quantified_decision"

ANALYSIS_START=2022-07-21   # 覆盖 训练3年 + 样本内2年 + 样本外1年
ANALYSIS_END=2026-07-21
BASE="backtest_result/exp3yr"
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR" "$BASE/analysis"
DATASET_CACHE="$ROOT/analysis/full_dataset_exp3yr.parquet"

XGB_PKL="$ROOT/models/exp3yr_xgb/latest/multi_objective_factor_model.pkl"
LGB_PKL="$ROOT/models/exp3yr_lgb/latest/multi_objective_factor_model.pkl"
NEU_PKL="$ROOT/models/exp3yr_neutral/latest_neural/neural_multi_objective_model.pkl"

# ── 等待主流程全部回测结束（neutral 样本外 by_objective 产出 metrics 即视为完成）──
SENTINEL_DIR="$ROOT/backtest_result/exp3yr/neutral/backtest2_out_of_sample/by_objective"
MAX_WAIT=28800   # 最多等 8 小时
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

run_one_analysis () {
  local TAG="$1"; local PKL="$2"; local EXTRA="$3"
  if [ ! -f "$PKL" ]; then
    echo "!! [$TAG] 模型文件缺失 ($PKL)，跳过分析"; return 1
  fi
  local OUT="$BASE/analysis/$TAG"
  echo "##### [$(date +%H:%M:%S)] ANALYSE $TAG -> $OUT"
  "$PY" -u -m analysis.run_analysis \
    --model "$PKL" --out "$OUT" \
    --start $ANALYSIS_START --end $ANALYSIS_END \
    --dataset "$DATASET_CACHE" --cache "$DATASET_CACHE" \
    $EXTRA 2>&1 | tee -a "$LOGDIR/analysis_${TAG}.log"
}

# 数据集只构建一次（首个模型构建并缓存，其余复用 --dataset）
run_one_analysis xgb    "$XGB_PKL" ""
run_one_analysis lgb    "$LGB_PKL" ""
run_one_analysis neutral "$NEU_PKL" "--no-shap"

echo "ALL EXP3YR ANALYSIS DONE $(date +%H:%M:%S)"
