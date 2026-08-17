#!/usr/bin/env bash
# T092：同折 NAM vs XGBoost vs LightGBM 的剩余种子。
# 用法：bash scripts/archive/t_series/run_t092_trees.sh <model>   （model = xgboost | lightgbm）
#
# 与 T090_base 严格可比：同缓存、同折、同族裁剪、同评估函数。
# 种子对树只影响 subsample/colsample 抽样，敏感度远低于 NAM，但仍跑满 4 个以便配对比较。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
MODEL="$1"
case "$MODEL" in xgboost) TAG=xgb;; lightgbm) TAG=lgb;; *) echo "未知模型 $MODEL"; exit 2;; esac

for SEED in 42 11 23 37; do
  JSON="diagnose_output/T092_${TAG}_s${SEED}.json"
  [ -f "$JSON" ] && { echo "[SKIP] $JSON"; continue; }
  echo "[TRAIN] $MODEL seed=$SEED  $(date '+%H:%M:%S')"
  "$PY" -u scripts/exp/exp_tree_vs_nam.py --model "$MODEL" \
    --stocks 800 --years 13 --end 2022-09-05 --drop-groups forecast \
    --cache-dir database/system_data/factors_cache_2026-08-14-fwdadjust \
    --folds 0.6:0.8,0.7:0.9,0.8:1.0 --allow-degenerate-downside-risk \
    --seed "$SEED" --output "$JSON" \
    > "diagnose_output/T092_${TAG}_s${SEED}_train.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL] seed=$SEED rc=$rc"; tail -15 "diagnose_output/T092_${TAG}_s${SEED}_train.log"; }
done
echo "[DONE] $MODEL $(date '+%H:%M:%S')"
"$PY" -u scripts/exp/analyze_multifold.py --prefix "T092_${TAG}" --base-prefix T090_base
