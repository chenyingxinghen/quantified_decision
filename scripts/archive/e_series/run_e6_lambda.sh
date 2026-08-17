#!/usr/bin/env bash
# T074: 把 T073 学到的门控敏感度整体缩放 a -> λ·a，找是否存在两窗口都不劣化的 λ*。
#
# 关键效率点：**λ 不需要重训**。a 是存档里的 11 个数，直接改写即可，
# λ=0 严格等价于基线（softmax 于全零 logits = 均匀权重 = --disable-gate）。
# 所以这一轮只有 8 次回测、零训练。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LAM="${1:-0.5}"
LTAG=$(echo "$LAM" | tr '.' 'p')
SEEDS="42 11 23 37"

LOCK="diagnose_output/.e6_lambda.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

for SEED in $SEEDS; do
  SRC="models/nam_gate/T073_e6_scalar_s${SEED}"
  DST="models/nam_gate/T074_e6_lam${LTAG}_s${SEED}"
  if [ ! -f "$DST/nam_gate_factor_model.pkl" ]; then
    echo "[SCALE] seed=$SEED λ=$LAM -> $DST"
    "$PY" scripts/archive/oneoff/scale_gate_sensitivity.py --src "$SRC" --dst "$DST" --lam "$LAM" || exit 1
  fi
done

bt_one () {
  SEED="$1"; WIN="$2"
  if [ "$WIN" = "bull" ]; then S=2024-08-05; E=2026-08-05; else S=2022-09-05; E=2024-08-05; fi
  MODEL="models/nam_gate/T074_e6_lam${LTAG}_s${SEED}"
  TAG="T074_lam${LTAG}_s${SEED}_k40_${WIN}"
  OUT="backtest_result/T074_e6_lam${LTAG}_s${SEED}/nam/conf0_${TAG}_minp1_nost_mp40_${S}_to_${E}"
  TRADES="${OUT}/backtest_trades.csv"
  if [ ! -f "$TRADES" ]; then
    echo "[BT ] $TAG"
    "$PY" scripts/run_backtest.py \
      --model "$MODEL" --start "$S" --end "$E" \
      --min-confidence 0 --risk-min-price 1 --exclude-st \
      --max-positions 40 --tag "$TAG" \
      > "diagnose_output/${TAG}_backtest.log" 2>&1
  fi
  [ -f "$TRADES" ] || { echo "[FAIL BT] $TAG"; return 1; }
  if [ ! -f "diagnose_output/random_null_${TAG}.json" ]; then
    echo "[NULL] $TAG"
    "$PY" scripts/exp/diag_random_null.py --trades "$TRADES" --n-sims 2000 \
      --buy-cost 0.0008 --sell-cost 0.0013 \
      --out "diagnose_output/random_null_${TAG}.json" \
      > "diagnose_output/${TAG}_random.log" 2>&1
  fi
}

for SEED in $SEEDS; do
  bt_one "$SEED" bear & P1=$!
  bt_one "$SEED" bull & P2=$!
  wait $P1 $P2
  echo "--- seed=$SEED 完成 ---"
done
echo "=== T074 λ=$LAM DONE ==="
