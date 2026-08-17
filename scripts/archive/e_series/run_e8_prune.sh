#!/usr/bin/env bash
# T077 / E8：**剪枝**（去掉机械生成的交叉族）—— 直接检验「瓶颈是噪声列数，不是信息」
#
# 立论来自 T076 的失败方式：E7 加了 18 列**理论上有信息**的行业内分位，IC 却 1/4、
# 中位 −0.0030。NAM 把所有列的贡献直接相加、没有选择机制，所以一列的边际效应 =
# 「信息增益 − 形状函数带来的噪声」。加信息列反而变差，说明当前边际是**负**的，
# 瓶颈在列数而不在信息量。219 列里有 76 列（35%）是机械生成的交叉项：
#   cross_fund 55（*_x_YOY* / *_div_{估值}）、cross_tech 21（*_mul_* / *_sub_*）。
# 它们与母因子高度共线，是最可能的纯噪声源（台账 group_effectiveness.csv 早有此迹象）。
#
# 两臂（各 4 种子，仅训练，0 回测；`--drop-groups` 是既有开关，零新代码）：
#   A: --drop-groups cross_fund              219 → 164
#   B: --drop-groups cross_fund,cross_tech   219 → 143
# 判定：scripts/exp/analyze_ic.py，晋级需 4/4 ΔIC>0 且中位 ≥ +0.005。
# 若 B 优于 A 优于基线，说明剪枝方向单调有效，下一轮继续往下剪并**在剪枝后的底座上
# 重测 E7**（稀释被去掉后，行业内分位可能才显出价值）。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e8_prune.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

train_arm () {
  TAG="$1"; DROP="$2"
  for SEED in 42 11 23 37; do
    JSON="diagnose_output/${TAG}_s${SEED}.json"
    OUT="models/nam_gate/${TAG}_s${SEED}"
    LOG="diagnose_output/${TAG}_s${SEED}_train.log"
    if [ -f "$JSON" ]; then echo "[SKIP] $JSON"; continue; fi
    echo "[TRAIN] $TAG seed=$SEED (drop=$DROP)"
    "$PY" scripts/exp/exp_nam_gate.py \
      --stocks 800 --years 13 --end 2022-09-05 \
      --disable-gate --target returns \
      --y-scale 2 --lambda-lb 0 \
      --epochs 60 --min-epochs 20 --skip-baseline \
      --folds "0.8:1.0" --seed "$SEED" \
      --allow-degenerate-downside-risk \
      --drop-groups "$DROP" \
      --save-model-dir "$OUT" \
      --output "$JSON" \
      > "$LOG" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] $TAG seed=$SEED rc=$rc"; tail -20 "$LOG"; }
  done
  "$PY" scripts/exp/analyze_ic.py --prefix "$TAG" --label "$TAG (drop=$DROP)" || true
}

train_arm T077_nocf  "cross_fund"
train_arm T077_nocfct "cross_fund,cross_tech"
echo "=== T077 E8 PRUNE DONE ==="
