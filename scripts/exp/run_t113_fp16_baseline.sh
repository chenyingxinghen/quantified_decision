#!/usr/bin/env bash
# T113：fp16 驻留的**加性基线批**（219 列 × 4 种子同进程）。
#
# 为什么要再跑一个基线，而不是直接拿 T098 当配对基线：
#   T114（瘦身）/T115（指数特征）/T116（标量门终检）三条轴的新臂都会走
#   --store-dtype fp16 + --seeds 同进程（T109/T110 定下的效率通道）。
#   T098 是 fp32 分进程跑的，dtype 差异实测 −0.00085（T110 等价性检验，n=1），
#   量级已接近瘦身轴"不显著劣化"判据的分辨率。与其在三次判定里各背一次
#   dtype 混杂，不如花一批算力把基线搬进同一通道：之后所有配对都是
#   同 dtype、同进程、同数据构建。
#
# 配方 = T098 冠军配置原样（run_t098_full.sh），仅两处不同：
#   --store-dtype fp16（显式，不依赖 auto 推断）
#   --seeds 42,11,23,37（同进程复用数据构建，省 3×11.5 分钟）
#
# 判据（预注册）：
#   本批不判任何晋级/关轴，只做两件事：
#   1. 与 T098 同种子配对差 |Δ| 应与 T110 测到的 dtype 效应同量级（~0.001）；
#      若某种子 |Δ| > 0.01，视为 fp16 通道异常，停下来查，不得继续用作基线。
#   2. 作为 T114/T115/T116 的配对基线存档。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"

"$PY" -u scripts/exp/exp_nam_gate.py \
  --stocks 6000 --years 13 --end 2022-09-05 --disable-gate --target returns \
  --skip-baseline --allow-degenerate-downside-risk --cache-dir "$CACHE" \
  --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
  --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
  --y-scale 2 --expert-hidden 16 --lr 2e-3 \
  --store-dtype fp16 --seeds 42,11,23,37 \
  --save-model-dir "models/nam_gate/T113_fp16" \
  --plot-dir "diagnose_output/nam_gate_T113" \
  --output "diagnose_output/T113_fp16.json"
rc=$?
echo "[T113] exit=$rc  $(date '+%H:%M:%S')"
exit $rc
