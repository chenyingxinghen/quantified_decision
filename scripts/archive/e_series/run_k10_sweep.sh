#!/usr/bin/env bash
# E4b (T072): K=10 —— 把 K 的扫描方向反过来。
#
# 立论：E4 测出超额 pp 随 K **单调衰减**（s42 牛市 K20/40/60 = 36.9/18.5/15.0，
# 熊市 K20/40 = 28.9/16.9），4 种子配对中位在双窗口也都是 K40 < K20。梯度指向 K 更小，
# 而 E4 只往大的方向扫过——这是一次方向性失误，K<20 从未在 NAM 上试过。
# 同时它与今日另两条结论同源：标签残差化（去暴露）、跨种子集成（去特异性）都劣化，
# alpha 集中在头部/特异性里，任何摊薄都在削弱它。K 减小是唯一的"反摊薄"操作。
#
# 反证须标注：K 变小 → 组合方差变大，z（对随机零假设的标准化）可能反而下降。
# 因此**两口径并列判定**：超额 pp（经济）与 z（统计），熊市为主判。
#
# 判据：同种子配对差 (K=10) − (K=20)，n=4，基线复用已有 K=20 零假设。
set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.k10_sweep.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

name_of () { case "$1" in 42) echo "T045" ;; *) echo "T068_seed$1" ;; esac; }
dir_of  () { case "$1" in 42) echo "models/nam_gate/T045" ;; *) echo "models/nam_gate/T068_seed$1" ;; esac; }

run_one () {
  SEED="$1"; WIN="$2"
  if [ "$WIN" = "bull" ]; then S=2024-08-05; E=2026-08-05; else S=2022-09-05; E=2024-08-05; fi
  MODEL=$(dir_of "$SEED"); NAME=$(name_of "$SEED")
  TAG="T072_s${SEED}_k10_${WIN}"
  OUT="backtest_result/${NAME}/nam/conf0_${TAG}_minp1_nost_mp10_${S}_to_${E}"
  TRADES="${OUT}/backtest_trades.csv"

  if [ ! -f "$TRADES" ]; then
    echo "[BT ] $TAG"
    "$PY" scripts/run_backtest.py \
      --model "$MODEL" --start "$S" --end "$E" \
      --min-confidence 0 --risk-min-price 1 --exclude-st \
      --max-positions 10 --tag "$TAG" \
      > "diagnose_output/${TAG}_backtest.log" 2>&1
  fi
  if [ ! -f "$TRADES" ]; then
    echo "[FAIL] $TAG 未产出，见 diagnose_output/${TAG}_backtest.log"; return 1
  fi
  if [ ! -f "diagnose_output/random_null_${TAG}.json" ]; then
    echo "[NULL] $TAG"
    "$PY" scripts/exp/diag_random_null.py \
      --trades "$TRADES" --n-sims 2000 \
      --buy-cost 0.0008 --sell-cost 0.0013 \
      --out "diagnose_output/random_null_${TAG}.json" \
      > "diagnose_output/${TAG}_random.log" 2>&1
  fi
}

# R2 之后每进程面板约 3.7 GB，并发 2 路（约 11 GB 峰值）安全。
for SEED in 42 11 23 37; do
  run_one "$SEED" bear &
  P1=$!
  run_one "$SEED" bull &
  P2=$!
  wait $P1 $P2
  echo "--- seed=$SEED 完成 ---"
done

echo "=== T072 K10 SWEEP DONE ==="
