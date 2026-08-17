#!/usr/bin/env bash
# T091_ss 单臂驱动（2026-08-14 16:5x 接力）
#
# 为什么另起一个脚本：T091_mz 前两个种子 Δ折均IC = −0.0317 / −0.0373（0/2 上升，
# 反向 12× MDE），按 run_t090_forecast_and_magnitude.sh 头部预注册的省钱规则直接关臂，
# 所以掐掉了原驱动（它会接着跑 mz 的 seed 23/37）。ss 是 E19 的**干净臂**必须跑完：
# mz 的标签头尾比 14.19 vs rank 7.20，崩塌可能来自头部集中度而非幅度信息本身；
# ss 的头尾比 7.23 ≈ rank 7.20，只有间距形状不同。
#
# 与原驱动**逐字节相同**的 COMMON/FOLDS/CACHE，否则和 T090_base 不可配对。
# 唯一改动：python 加 -u（原驱动漏了，日志块缓冲会假死，见 [[bg-training-log-buffering-not-death]]）。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
FOLDS="0.6:0.8,0.7:0.9,0.8:1.0"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
COMMON="--stocks 800 --years 13 --end 2022-09-05 --disable-gate --target returns
        --y-scale 2 --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline
        --allow-degenerate-downside-risk --cache-dir $CACHE"

for SEED in 42 11 23 37; do
  JSON="diagnose_output/T091_ss_s${SEED}.json"
  [ -f "$JSON" ] && { echo "[SKIP] $JSON"; continue; }
  echo "[TRAIN] T091_ss seed=$SEED  $(date '+%H:%M:%S')"
  "$PY" -u scripts/exp/exp_nam_gate.py $COMMON --folds "$FOLDS" --seed "$SEED" \
    --drop-groups forecast --label-transform signed_sqrt \
    --output "$JSON" > "diagnose_output/T091_ss_s${SEED}_train.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL] seed=$SEED rc=$rc"; tail -20 "diagnose_output/T091_ss_s${SEED}_train.log"; }
done

echo "[DONE] $(date '+%H:%M:%S')"
"$PY" -u scripts/exp/analyze_multifold.py --prefix T091_ss --base-prefix T090_base
