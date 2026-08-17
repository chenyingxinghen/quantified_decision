#!/usr/bin/env bash
# T104：T102 的 both 臂（门控 + 显式交互列）扩到 4 种子做正式配对判定。
# 单种子 Δ 中位 +0.01048 过了预注册的 +0.005 扩种子门槛，但逐折离散很大
# (-0.00167/+0.01916/+0.01048)，超加性也可能是单种子噪声 —— 必须配对验证。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
COMMON="--stocks 800 --years 13 --end 2022-09-05 --target returns --skip-baseline \
--allow-degenerate-downside-risk --cache-dir $CACHE \
--folds 0.6:0.8,0.7:0.9,0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
--epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 --y-scale 2 \
--gate-mode softmax --keep-manual-interaction"
for S in 11 23 37; do
  J="diagnose_output/T102_both_s${S}.json"
  [ -f "$J" ] && { echo "[SKIP] $J"; continue; }
  echo "[TRAIN] both seed=$S  $(date '+%H:%M:%S')"
  "$PY" -u scripts/exp/exp_nam_gate.py $COMMON --seed "$S" --output "$J" \
    > "diagnose_output/T102_both_s${S}.log" 2>&1
  echo "[OK] s$S  $(date '+%H:%M:%S')"
done
echo "[DONE] $(date '+%H:%M:%S')"
