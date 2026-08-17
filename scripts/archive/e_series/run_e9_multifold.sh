#!/usr/bin/env bash
# T078 / E9：**先修评估器，再跑轴** —— 多折验证 IC 标定
#
# 为什么这是当前最高价值的一步（而不是再试一个特征想法）：
#   E7（+18 列行业内分位）Δ中位 −0.0030、1/4；
#   E8-A（−55 列 cross_fund）Δ中位 −0.0028、1/4。
#   **两个方向相反的改动给出几乎相同的「损失」，且都符号不一致**——这是纯种子噪声的
#   指纹，不是效应。也就是说单折（80-100%）验证 Rank IC 的噪声带就有 ±0.003~0.004，
#   而我们设的晋级门槛是 +0.005：任何真实的中小效应都被埋在噪声里，继续换特征只是
#   在噪声里抽签。必须先把评估器的分辨率提上去。
#
# 做法（仍然零回测，且几乎免费）：`--folds` 支持多折，而**数据加载只做一次**，
# 所以 3 折 ≈ 1.2× 单折成本，验证天数 ×3 → IC 标准误 ≈ /√3。
#   --folds "0.6:0.8,0.7:0.9,0.8:1.0"
# 本轮只跑**基线**（--disable-gate，不动特征），目的是量出：
#   1) 每折 IC 与折间一致性（80-100% 是不是一个特殊窗口）
#   2) 折均 IC 的跨种子 σ_seed → 新的 IC 判定 MDE
# 之后所有打分层轴都用「折均 IC + 新 MDE」判定，并按新口径重判 E7/E8。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e9_multifold.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
# 等 E8 跑完，避免两个训练进程抢内存（08-08 事故）
for i in $(seq 1 240); do
  if [ -f diagnose_output/.e8_prune.lock ]; then
    old=$(cat diagnose_output/.e8_prune.lock 2>/dev/null)
    if ps -p "$old" >/dev/null 2>&1; then sleep 30; continue; fi
  fi
  break
done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

FOLDS="0.6:0.8,0.7:0.9,0.8:1.0"
for SEED in 42 11 23 37; do
  JSON="diagnose_output/T078_mf_base_s${SEED}.json"
  LOG="diagnose_output/T078_mf_base_s${SEED}_train.log"
  if [ -f "$JSON" ]; then echo "[SKIP] $JSON"; continue; fi
  echo "[TRAIN] 多折基线 seed=$SEED folds=$FOLDS"
  "$PY" scripts/exp/exp_nam_gate.py \
    --stocks 800 --years 13 --end 2022-09-05 \
    --disable-gate --target returns \
    --y-scale 2 --lambda-lb 0 \
    --epochs 60 --min-epochs 20 --skip-baseline \
    --folds "$FOLDS" --seed "$SEED" \
    --allow-degenerate-downside-risk \
    --output "$JSON" \
    > "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL] seed=$SEED rc=$rc"; tail -20 "$LOG"; }
done
echo "=== T078 E9 MULTIFOLD BASELINE DONE ==="
"$PY" scripts/exp/analyze_multifold.py --prefix T078_mf_base || true
