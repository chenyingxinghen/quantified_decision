#!/usr/bin/env bash
# T086 / E16：训练样本的**时间衰减权重** —— 剩下唯一没试过、且不被阻塞的打分层杠杆
#
# 立论：训练跨 13 年（2009→2022），A 股的结构（涨跌停/注册制/北向/量化占比）变过多次，
# 久远样本可能已经是负资产。但下调老样本同时减少有效样本量 —— 方向不能靠拍脑袋，
# 所以扫两个 τ 看斜率，而不是只试一个点：
#   τ=3y  激进（13 年前的日权重 e^-4.33≈0.013）
#   τ=7y  温和（e^-1.86≈0.156）
# 两个都不涨 ⇒ 时间轴关闭，且能确认「不是 τ 选错了」。
#
# 判定：T082 双条件门槛
#   ① 合并 Δ折均IC ≥ +0.0028（2×MDE）且 4/4 同向；② 下跌日 ΔIC 不一边倒为负。
# 基线复用 T083_base（已带 regime 段），所以本轮只需 8 次训练、零回测。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e16.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

FOLDS="0.6:0.8,0.7:0.9,0.8:1.0"
COMMON="--stocks 800 --years 13 --end 2022-09-05 --disable-gate --target returns
        --y-scale 2 --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline
        --allow-degenerate-downside-risk"

run_arm () {
  TAG="$1"; shift
  for SEED in 42 11 23 37; do
    JSON="diagnose_output/${TAG}_s${SEED}.json"
    [ -f "$JSON" ] && { echo "[SKIP] $JSON"; continue; }
    echo "[TRAIN] $TAG seed=$SEED"
    "$PY" scripts/exp/exp_nam_gate.py $COMMON --folds "$FOLDS" --seed "$SEED" "$@" \
      --output "$JSON" > "diagnose_output/${TAG}_s${SEED}_train.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] $TAG seed=$SEED rc=$rc"; tail -20 "diagnose_output/${TAG}_s${SEED}_train.log"; }
  done
}

run_arm T086_td3 --time-decay-years 3
run_arm T086_td7 --time-decay-years 7

echo "=== T086 E16 DONE ==="
for T in T086_td3 T086_td7; do
  echo "########## $T ##########"
  "$PY" scripts/exp/analyze_multifold.py --prefix "$T" --base-prefix T083_base || true
done
