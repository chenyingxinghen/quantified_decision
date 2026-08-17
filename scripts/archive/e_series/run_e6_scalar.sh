#!/usr/bin/env bash
# T073 / E6：标量条件化门控（唯一还没关的既有轴）
#
# 立论（台账 E6 节）：T046 的 oracle IC +128% 说明条件结构确实存在，但可实现增益≈0，
# 正确解释不是「结构不存在」，而是在约 90 个独立块上学不动 d_regime→13族 的 (64,32) MLP
# （2000+ 参数）。本轮把门控参数量从 2000+ 降到 **13**（每族一个敏感度标量）：
#   logits = a · (2·vol_expand − 1)，  w = 13 · softmax(logits)
# a=0 初始化 → 起点与基线（--disable-gate 的均匀权重）完全一致，所以任何偏离都是学到的。
#
# 变量隔离：与 T045/T068 基线**唯一**差别是 --disable-gate → --gate-mode scalar。
# 判定口径（新协议，见台账 E4 判定结果）：**在 K=40 上筛选**（跨种子收益 std 只有
# K=20 的 1/2.7，熊市 MDE 1.64 z），同种子配对 vs 已有的 T069_s*_k40 零假设，n=4。
# 若熊市 Δz 中位 > +1.64 且不是 1/4 正向，才回 K=20 做最终确认。
set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e6_scalar.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

COL="vol_expand"
SEEDS="42 11 23 37"

# ---- 1) 训练（串行：训练是 GPU/CPU 密集，且回测互斥检查会看这个进程名）----
for SEED in $SEEDS; do
  OUT="models/nam_gate/T073_e6_scalar_s${SEED}"
  LOG="diagnose_output/T073_e6_scalar_s${SEED}_train.log"
  if [ -f "$OUT/nam_gate_factor_model.pkl" ]; then
    echo "[SKIP TRAIN] $OUT 已存在"; continue
  fi
  echo "[TRAIN] seed=$SEED -> $OUT"
  "$PY" scripts/exp/exp_nam_gate.py \
    --stocks 800 --years 13 --end 2022-09-05 \
    --gate-mode scalar --gate-scalar-col "$COL" \
    --target returns \
    --y-scale 2 --lambda-lb 0 \
    --epochs 60 --min-epochs 20 --skip-baseline \
    --folds "0.8:1.0" --seed "$SEED" \
    --allow-degenerate-downside-risk \
    --save-model-dir "$OUT" \
    --output "diagnose_output/T073_e6_scalar_s${SEED}.json" \
    > "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[FAIL TRAIN] seed=$SEED rc=$rc 见 $LOG"; tail -20 "$LOG"; fi
  # T043 铁律：模型目录必须有 norm_stats.pkl，否则回测会以原始量纲喂特征
  if [ ! -f "$OUT/norm_stats.pkl" ]; then
    echo "[FAIL] $OUT 缺 norm_stats.pkl -> 该种子不可用于回测"
  fi
done

# ---- 2) 回测 + 随机零假设（K=40 筛选口径，双窗口，2 路并行）----
bt_one () {
  SEED="$1"; WIN="$2"
  if [ "$WIN" = "bull" ]; then S=2024-08-05; E=2026-08-05; else S=2022-09-05; E=2024-08-05; fi
  MODEL="models/nam_gate/T073_e6_scalar_s${SEED}"
  [ -f "$MODEL/norm_stats.pkl" ] || { echo "[SKIP BT] $MODEL 不可用"; return 1; }
  TAG="T073_e6_s${SEED}_k40_${WIN}"
  OUT="backtest_result/T073_e6_scalar_s${SEED}/nam/conf0_${TAG}_minp1_nost_mp40_${S}_to_${E}"
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
    echo "[FAIL BT] $TAG 见 diagnose_output/${TAG}_backtest.log"; return 1
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

for SEED in $SEEDS; do
  bt_one "$SEED" bear & P1=$!
  bt_one "$SEED" bull & P2=$!
  wait $P1 $P2
  echo "--- seed=$SEED 回测完成 ---"
done

echo "=== T073 E6 SCALAR DONE ==="
