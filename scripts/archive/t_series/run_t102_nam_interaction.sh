#!/usr/bin/env bash
# T102：NAM 补两个「学交互」的臂 —— 单种子，800 只（与 T096_base_s42 严格同配置对照）。
#
# **为什么重开这条轴**：同面板同尺子下 lgb 比 NAM 高 0.0077 = 13× 配对 MDE（3/3 种子），
# 而 NAM 生产配置是 `--disable-gate` 的**纯加性**模型，结构上学不了任何交互。
# 树多拿的那部分很可能就是交互项 —— 值得给 NAM 两次机会：
#
#   gate : 打开 softmax 门控（29 维 regime → 11 个族权重）。这是「按市场状态调
#          各族权重」的乘性交互。⚠ 必须用修复后的 RegimeGate —— 2026-08-15 修掉了
#          scalar 分支的双重中心化（复合成 4r−3），E6 的历史判定建立在畸变实现上。
#   xint : 保留 8 个手工 *_regime_* 交互列当显式特征（平时为 NAM 剔除，因为它学不了
#          交互所以本该由门控代劳；现在反过来直接喂给它）。
#
# 判据：对 T096_base_s42 同折 holdout。单种子没有配对功率，只看**方向和量级**：
#   Δ ≥ +0.005（≈ lgb 领先量的 2/3）才值得扩到多种子；|Δ| < 0.002 直接关轴。
# 成本：每臂 3 折约 25 分钟。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
COMMON="--stocks 800 --years 13 --end 2022-09-05 --target returns --skip-baseline \
--allow-degenerate-downside-risk --cache-dir $CACHE \
--folds 0.6:0.8,0.7:0.9,0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
--epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 --y-scale 2 --seed 42"

run () {  # run <标签> <额外参数...>
  local tag="$1"; shift
  local J="diagnose_output/T102_${tag}_s42.json"
  [ -f "$J" ] && { echo "[SKIP] $J"; return; }
  echo "[TRAIN] $tag  $(date '+%H:%M:%S')"
  "$PY" -u scripts/exp/exp_nam_gate.py $COMMON "$@" --output "$J" \
    > "diagnose_output/T102_${tag}_s42.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL] $tag rc=$rc"; tail -12 "diagnose_output/T102_${tag}_s42.log"; }
  echo "[OK] $tag  $(date '+%H:%M:%S')"
}

# 门控打开：不传 --disable-gate 即启用 softmax 门控
run gate --gate-mode softmax
# 显式交互列：仍关门控，隔离「特征侧交互」的单独贡献
run xint --disable-gate --keep-manual-interaction
# 两者叠加
run both --gate-mode softmax --keep-manual-interaction

echo "[DONE] $(date '+%H:%M:%S')"
