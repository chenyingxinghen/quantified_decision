#!/usr/bin/env bash
# T109 基准：全量池 + 门控（T108 同配置），fp16 驻留 vs fp32 驻留，各 4 epoch。
# 只测速度，不看 IC —— epoch 数不足以判效果。
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
COMMON="--stocks 6000 --years 13 --end 2022-09-05 --target returns --skip-baseline \
--allow-degenerate-downside-risk --cache-dir $CACHE \
--folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
--epochs 4 --min-epochs 1 --y-scale 2 --expert-hidden 16 --lr 2e-3 \
--gate-mode softmax --lambda-lb 0.01 --seed 11"
for DT in fp16 fp32; do
  echo "=== $DT  $(date '+%H:%M:%S') ==="
  "$PY" -u scripts/exp/stamp.py \
    "$PY" -u scripts/exp/exp_nam_gate.py $COMMON --store-dtype "$DT" \
    --output "diagnose_output/T109_bench_${DT}.json" \
    > "diagnose_output/T109_bench_${DT}.log" 2>&1
  echo "  rc=$? $(date '+%H:%M:%S')"
done
touch diagnose_output/.t109_bench.done
echo "[DONE] $(date '+%H:%M:%S')"
