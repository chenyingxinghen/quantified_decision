#!/usr/bin/env bash
# T081 / E11 确认轮：把 T080 晋级的多持有期标签放到**回测**上验证。
#
# T080 是 IC 晋级（4/4，Δ折均IC +0.00296 ≥ 2×MDE）。按新协议，只有 IC 过门槛的改动
# 才花回测。这一轮做两件事：
#   1) 按基线可回测口径重训（--folds "0.8:1.0" + --save-model-dir）。
#      T080 用的是 3 折评估口径、且没存模型；回测必须用与 T045/T068 基线同口径的单折模型。
#   2) K=40 双窗口回测 + 随机零假设，同种子配对 vs random_null_T069_s{seed}_k40_{win}。
#      熊市为主判（MDE 1.64 z），牛市 MDE 1.13 z。
#
# 注意：多持有期**只改训练标签**，推理/回测路径不需要任何改动 —— 所以回测脚本和
# 基线完全一致，唯一差别在模型权重。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e11_confirm.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
for i in $(seq 1 240); do
  if [ -f diagnose_output/.e11_mh.lock ]; then
    old=$(cat diagnose_output/.e11_mh.lock 2>/dev/null)
    if ps -p "$old" >/dev/null 2>&1; then sleep 30; continue; fi
  fi
  break
done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

SEEDS="42 11 23 37"

# ---- 1) 训练（单折口径，存模型）----
for SEED in $SEEDS; do
  OUT="models/nam_gate/T081_mh_s${SEED}"
  LOG="diagnose_output/T081_mh_s${SEED}_train.log"
  if [ -f "$OUT/nam_gate_factor_model.pkl" ]; then echo "[SKIP TRAIN] $OUT"; continue; fi
  echo "[TRAIN] seed=$SEED -> $OUT"
  "$PY" scripts/exp/exp_nam_gate.py \
    --stocks 800 --years 13 --end 2022-09-05 \
    --disable-gate --target returns \
    --y-scale 2 --lambda-lb 0 \
    --epochs 60 --min-epochs 20 --skip-baseline \
    --folds "0.8:1.0" --seed "$SEED" \
    --allow-degenerate-downside-risk \
    --multi-horizon 5,10,20 \
    --save-model-dir "$OUT" \
    --output "diagnose_output/T081_mh_s${SEED}.json" \
    > "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL TRAIN] seed=$SEED rc=$rc"; tail -20 "$LOG"; }
  # T043 铁律
  [ -f "$OUT/norm_stats.pkl" ] || echo "[FAIL] $OUT 缺 norm_stats.pkl，不可回测"
done

# ---- 2) K=40 双窗口回测 + 零假设（2 路并行）----
bt_one () {
  SEED="$1"; WIN="$2"
  if [ "$WIN" = "bull" ]; then S=2024-08-05; E=2026-08-05; else S=2022-09-05; E=2024-08-05; fi
  MODEL="models/nam_gate/T081_mh_s${SEED}"
  [ -f "$MODEL/norm_stats.pkl" ] || { echo "[SKIP BT] $MODEL 不可用"; return 1; }
  TAG="T081_mh_s${SEED}_k40_${WIN}"
  OUT="backtest_result/T081_mh_s${SEED}/nam/conf0_${TAG}_minp1_nost_mp40_${S}_to_${E}"
  TRADES="${OUT}/backtest_trades.csv"
  if [ ! -f "$TRADES" ]; then
    echo "[BT ] $TAG"
    "$PY" scripts/run_backtest.py \
      --model "$MODEL" --start "$S" --end "$E" \
      --min-confidence 0 --risk-min-price 1 --exclude-st \
      --max-positions 40 --tag "$TAG" \
      > "diagnose_output/${TAG}_backtest.log" 2>&1
  fi
  if [ ! -f "$TRADES" ]; then
    actual=$(ls -d backtest_result/T081_mh_s${SEED}/nam/*${TAG}* 2>/dev/null | head -1)
    if [ -n "$actual" ]; then
      echo "[口径核对] 期望 $OUT"; echo "           实得 $actual"
      TRADES="${actual}/backtest_trades.csv"
      case "$actual" in *_nost_mp40_*) : ;; *) echo "[拒绝] 缺 nost/mp40 段"; return 1 ;; esac
    else
      echo "[FAIL BT] $TAG 见 diagnose_output/${TAG}_backtest.log"; return 1
    fi
  fi
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
echo "=== T081 E11 CONFIRM DONE ==="
"$PY" scripts/archive/e_series/analyze_e11_bt.py || true
