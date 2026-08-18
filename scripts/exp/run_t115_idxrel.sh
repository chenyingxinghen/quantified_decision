#!/usr/bin/env bash
# T115：指数相对特征臂（219 + 5 = 224 列，新族 index_rel）× 4 种子。
#
# 前置：database/system_data/factors_cache_2026-08-18-idxrel 已由
#       scripts/build_idxrel_cache.py 构建完成（约 40~60 分钟 CPU）。
#
# 预注册判据（这是晋级轴，按 IC 优先协议）：
#   晋级门：配对 Δ(T115−T113) holdout IC 4/4 为正且 Δ中位 ≥ 0.005。
#   否决门（regime-stratified）：跌日 ΔIC ≤1/3 种子为正 → 关轴，
#           无论整体 Δ 多好看（防「涨日赢跌日输」被平均成净正）。
#   落在中间（方向正但不够 0.005）→ 记录为"未达晋级、值得跟踪"，不进生产。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-18-idxrel"

if [ ! -f "$CACHE/factor_cache_manifest.json" ]; then
  echo "[ABORT] 新缓存未构建完成（缺 $CACHE/factor_cache_manifest.json），先跑:"
  echo "  $PY -u scripts/build_idxrel_cache.py"
  exit 2
fi

"$PY" -u scripts/exp/exp_nam_gate.py \
  --stocks 6000 --years 13 --end 2022-09-05 --disable-gate --target returns \
  --skip-baseline --allow-degenerate-downside-risk --cache-dir "$CACHE" \
  --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
  --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
  --y-scale 2 --expert-hidden 16 --lr 2e-3 \
  --store-dtype fp16 --seeds 42,11,23,37 \
  --save-model-dir "models/nam_gate/T115_idxrel" \
  --plot-dir "diagnose_output/nam_gate_T115" \
  --output "diagnose_output/T115_idxrel.json"
rc=$?
echo "[T115] exit=$rc  $(date '+%H:%M:%S')"
exit $rc
