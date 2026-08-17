#!/usr/bin/env bash
# T090 / E18 收尾 + T091 / E19：**重建后的新基线** → 预告因子组 → 幅度标签
#
# 为什么必须先重新建基线：本轮把价格复权从「稀疏 adjust_factor + bfill/ffill」
# 换成了 preclose/close 累乘的完整前复权（见 core/factors/train_ml_model.py::
# compute_forward_adjust_factor）。它同时改了**因子**和**标签**——7 日前向收益
# 直接由 close 算出——所以台账里 T083 及之前所有绝对 IC 数值不再可比。
# 任何 Δ 判定都必须对同一份新缓存下的新基线做，否则把复权修复的效应和
# 新轴的效应混在一起。
#
# 三个臂（全部 4 种子 × 3 折，零回测）：
#   T090_base  新基线：219 列，不含预告                  ← 新的比较基准
#   T090_fc    +4 列预告（forecast 族）                  ← E18 的最后一个问题
#   T091_mz    winsor_z 幅度标签                          ← E19
#   T091_ss    signed_sqrt 幅度标签                       ← E19（厚尾对照）
#
# 判定（G4 双门槛，见 TRAINING_ITERATIONS.md 与 [[regime-stratified-ic-gate]]）：
#   ① 合并 Δ折均 IC ≥ +0.0028（2×MDE）且 4/4 种子同向；
#   ② 下跌日 ΔIC **不得**一边倒为负（≤1/3 为正即判 regime 倾斜，直接关轴）。
# 两个条件都过才谈回测；只过①是 E11/E16 那种「验证 IC 涨、OOS 熊市掉 9pp」的陷阱。
#
# 成本：每次训练约 28 分钟（RTX 4050），16 次 ≈ 7.5 小时。
# 省钱策略：种子顺序是 42,11,23,37 —— 先看前两个，若 Δ 与门槛差一个数量级
# 就直接 Ctrl-C 关轴，不必烧满 4 个。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.t090.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

FOLDS="0.6:0.8,0.7:0.9,0.8:1.0"
# 新公式必须用独立缓存目录（config/factor_config.py 的规矩：禁止原地覆盖旧缓存，
# 否则历史模型的推理输入会静默漂移）。这个目录由
#   scripts/train_model.py --update-cache-only --force --stocks 6000 --cache-dir <此路径>
# 重建，manifest 版本 = 2026-08-14-fwdadjust-preclose-forecast-v1。
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
COMMON="--stocks 800 --years 13 --end 2022-09-05 --disable-gate --target returns
        --y-scale 2 --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline
        --allow-degenerate-downside-risk --cache-dir $CACHE"

run_arm () {
  TAG="$1"; shift
  for SEED in 42 11 23 37; do
    JSON="diagnose_output/${TAG}_s${SEED}.json"
    [ -f "$JSON" ] && { echo "[SKIP] $JSON"; continue; }
    echo "[TRAIN] $TAG seed=$SEED  $(date '+%H:%M:%S')"
    "$PY" scripts/exp/exp_nam_gate.py $COMMON --folds "$FOLDS" --seed "$SEED" "$@" \
      --output "$JSON" > "diagnose_output/${TAG}_s${SEED}_train.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] $TAG seed=$SEED rc=$rc"; tail -20 "diagnose_output/${TAG}_s${SEED}_train.log"; }
  done
}

# 新基线。--drop-groups forecast 把 4 个预告列从面板里剔掉，保证 base 与 fc 臂
# 的差别**只有**这 4 列（而不是「重建前 vs 重建后」两件事混在一起）。
run_arm T090_base --drop-groups forecast
run_arm T090_fc
run_arm T091_mz --drop-groups forecast --label-transform winsor_z
run_arm T091_ss --drop-groups forecast --label-transform signed_sqrt

echo "=== T090/T091 DONE $(date '+%H:%M:%S') ==="
for T in T090_fc T091_mz T091_ss; do
  echo "########## $T vs T090_base ##########"
  "$PY" scripts/exp/analyze_multifold.py --prefix "$T" --base-prefix T090_base || true
done
