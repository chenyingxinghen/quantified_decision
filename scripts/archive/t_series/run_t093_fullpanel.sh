#!/usr/bin/env bash
# T093：树吃满全 panel（231 列）vs T092 的受限口径（219 列）vs NAM T090_base。
#
# T092 为了单变量可比，把树限制在 NAM 的特征集 —— 剔了 8 列手工 *_regime_* 交互
# 和 4 列 forecast 族。但手工交互本来就是**为 NAM 造的**（它严格可加、学不了交互）��
# 对能自学交互的树是双重惩罚。这一轮让树满配，回答「各自满配时哪个模型更强」。
#
# 三组读数的用法：
#   T093 − T092（同模型、同种子）= 那 12 列对树值多少
#   T093 − T090_base             = 各自满配的选型对照
# 无偏横比仍看 rank_ic_fixed（固定 300 棵，不做验证集选型）。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
MODEL="$1"
case "$MODEL" in xgboost) TAG=xgb;; lightgbm) TAG=lgb;; *) echo "未知模型 $MODEL"; exit 2;; esac

for SEED in 42 11 23 37; do
  JSON="diagnose_output/T093_${TAG}_s${SEED}.json"
  [ -f "$JSON" ] && { echo "[SKIP] $JSON"; continue; }
  echo "[TRAIN] $MODEL seed=$SEED  $(date '+%H:%M:%S')"
  "$PY" -u scripts/exp/exp_tree_vs_nam.py --model "$MODEL" \
    --stocks 800 --years 13 --end 2022-09-05 --full-panel \
    --cache-dir database/system_data/factors_cache_2026-08-14-fwdadjust \
    --folds 0.6:0.8,0.7:0.9,0.8:1.0 --allow-degenerate-downside-risk \
    --seed "$SEED" --output "$JSON" \
    > "diagnose_output/T093_${TAG}_s${SEED}_train.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL] seed=$SEED rc=$rc"; tail -15 "diagnose_output/T093_${TAG}_s${SEED}_train.log"; }
done
echo "[DONE] $MODEL $(date '+%H:%M:%S')"
"$PY" -u scripts/exp/analyze_multifold.py --prefix "T093_${TAG}" --base-prefix T090_base
"$PY" -u scripts/exp/analyze_multifold.py --prefix "T093_${TAG}" --base-prefix "T092_${TAG}"
