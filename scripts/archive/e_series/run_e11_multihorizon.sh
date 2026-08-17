#!/usr/bin/env bash
# T080 / E11：**多持有期复合标签**（降标签噪声）—— 用 T078 标定过的多折评估器判定
#
# 立论：E8 已证明特征侧到顶（删列单调劣化、加派生列无增益 ⇒ 信息受限），而库外新信息
# （业绩预告，E10）被 baostock 黑名单阻塞。剩下唯一还没试过、且不需要新数据的 alpha 杠杆
# 是**标签噪声**：单一 7 日前向收益里绝大部分是不可预测的价格噪声，7 日 IC 的天花板
# 很大程度是标签方差而不是模型能力。多持有期（5/10/20d）各自做当日截面 rank 再等权平均，
# 噪声约按 √k 衰减而共同的截面信号保留 —— 这是在**同一批数据**上提高信噪比，
# 不是再派生一列特征。
#
# 严格性三条：
#   1) **只换训练标签**，验证标签仍是 7 日基线口径 ⇒ val Rank IC 与 T078 基线严格可比。
#      要回答的是「训练目标噪声更小，能不能更准地预测同一个 7 日结果」。
#   2) 先 rank 再平均（不同持有期收益量纲差数倍，直接平均会被 20 日主导）。
#   3) 禁运期：训练标签最长看 20 天，切折只留 7 天间隔，差额 13 个交易日从训练集尾部
#      砍掉，否则 20 日标签会穿进验证窗口（脚本内已实现并打印实际砍掉的天数）。
#
# 判定：多折口径（T078 标定，MDE ≈ 0.0014），
#   python scripts/exp/analyze_multifold.py --prefix T080_mh5_10_20 --base-prefix T078_mf_base
#   晋级需 4/4 同向且 Δ折均IC ≥ +0.0028（2×MDE）。零回测。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e11_mh.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
# 串行等 E9 多折基线跑完（同时两个训练会把内存打爆，见 08-08 事故）
for i in $(seq 1 240); do
  if [ -f diagnose_output/.e9_multifold.lock ]; then
    old=$(cat diagnose_output/.e9_multifold.lock 2>/dev/null)
    if ps -p "$old" >/dev/null 2>&1; then sleep 30; continue; fi
  fi
  break
done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

TAG="T080_mh5_10_20"
FOLDS="0.6:0.8,0.7:0.9,0.8:1.0"
for SEED in 42 11 23 37; do
  JSON="diagnose_output/${TAG}_s${SEED}.json"
  LOG="diagnose_output/${TAG}_s${SEED}_train.log"
  if [ -f "$JSON" ]; then echo "[SKIP] $JSON"; continue; fi
  echo "[TRAIN] $TAG seed=$SEED"
  "$PY" scripts/exp/exp_nam_gate.py \
    --stocks 800 --years 13 --end 2022-09-05 \
    --disable-gate --target returns \
    --y-scale 2 --lambda-lb 0 \
    --epochs 60 --min-epochs 20 --skip-baseline \
    --folds "$FOLDS" --seed "$SEED" \
    --allow-degenerate-downside-risk \
    --multi-horizon 5,10,20 \
    --output "$JSON" \
    > "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL] seed=$SEED rc=$rc"; tail -20 "$LOG"; }
done
echo "=== T080 E11 MULTI-HORIZON DONE ==="
"$PY" scripts/exp/analyze_multifold.py --prefix "$TAG" --base-prefix T078_mf_base || true
