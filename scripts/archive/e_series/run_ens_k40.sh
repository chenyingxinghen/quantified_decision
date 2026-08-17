#!/usr/bin/env bash
# T071: 在 K=40（低噪评估口径）上重判「多种子横截面分位集成」轴。
#
# 为什么重跑一个已经跑过的东西：T068 在 K=20 上给出 |Δ|/σ_seed = 0.27，属**不可测量**，
# 既不能判劣也不能判优（台账已更正为「未测定」）。E4 刚测出 K=40 把跨种子 z_std 压到
# 约一半（熊 1.50→0.83、牛 1.19→0.57），非配对 n=4 的 MDE 从约 2.3 z 降到 1.1~1.6 z。
# 也就是说：同一个问题在 K=40 上**才有可能被回答**，而成本只有 2 次回测。
#
# 判据：ens7 的 z vs 4 个单种子 {42,11,23,37} 在 K=40 上的 z 中位（非配对，n=4）。
# 熊市为主判。门禁：Δz > +1.64（熊市 MDE）才算集成轴有效。
#
# 成员必须与 T068 完全一致（目录名里的 ens7-231379ba 哈希应复现，否则口径不同）。
set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1

# 见台账「驱动层踩坑」：stdout 重定向到文件时必须显式 UTF-8。
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.ens_k40.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then
    echo "[LOCK] 已有实例在跑 (pid=$old)"; exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

BASE="models/nam_gate/T045"
MEMBERS="models/nam_gate/T068_seed11 models/nam_gate/T068_seed23 models/nam_gate/T068_seed37 models/nam_gate/T068_seed53 models/nam_gate/T068_seed67 models/nam_gate/T068_seed89"

ENS_ARGS=""
for m in $MEMBERS; do
  if [ ! -f "${m}/nam_gate_factor_model.pkl" ]; then
    echo "[FAIL] 成员缺失: $m"; exit 1
  fi
  ENS_ARGS="$ENS_ARGS --ensemble-model $m"
done

run_one () {
  WIN="$1"; S="$2"; E="$3"
  TAG="T071_ens7_k40_${WIN}"
  LOG="diagnose_output/${TAG}_backtest.log"
  echo "[BT ] $TAG"
  "$PY" scripts/run_backtest.py \
    --model "$BASE" $ENS_ARGS \
    --start "$S" --end "$E" \
    --min-confidence 0 --risk-min-price 1 --exclude-st \
    --max-positions 40 --tag "$TAG" > "$LOG" 2>&1

  TRADES=$(ls -d backtest_result/T045/nam/*${TAG}*/backtest_trades.csv 2>/dev/null | head -1)
  if [ -z "$TRADES" ]; then
    echo "[FAIL] $TAG 未产出，见 $LOG"; return 1
  fi
  case "$TRADES" in
    *_nost_mp40_*) : ;;
    *) echo "[口径错配] $TRADES 缺 nost/mp40 段 -> 拒绝纳入判定"; return 1 ;;
  esac
  echo "[NULL] $TAG"
  "$PY" scripts/exp/diag_random_null.py \
    --trades "$TRADES" --n-sims 2000 \
    --buy-cost 0.0008 --sell-cost 0.0013 \
    --out "diagnose_output/random_null_${TAG}.json" \
    > "diagnose_output/${TAG}_random.log" 2>&1
}

# R2 之后单进程面板约 3.7 GB，两个窗口并行是安全的（两者共约 11 GB 峰值）。
run_one bear 2022-09-05 2024-08-05 &
P1=$!
run_one bull 2024-08-05 2026-08-05 &
P2=$!
wait $P1 $P2

echo "=== T071 ENS K40 DONE ==="
