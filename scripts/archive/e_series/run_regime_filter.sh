#!/usr/bin/env bash
# T075: 风控/择时层第一轴 —— --regime-filter trend（零训练）
#
# 为什么这是当前最高期望价值（见台账「今日效应量分层」）：
# 今天五个实验里，所有"把股票排得更准"的打分层改动效应量都 ≤ MDE（K 轴 |Δz|≤0.6、
# 集成 −1.18、标签残差化虽显著但为负），唯一远超 MDE 且 8/8 符号一致的效应来自
# T073 的门控，而它的验证 IC 是 0/4 **下降**的——它没排得更准，赢在暴露/时机。
# 结论：杠杆在组合层。T073 等于用错的位置（打分族权重）做了对的事（vol 条件化暴露）。
# `--regime-filter trend` 正是把同一件事放回它该在的位置：合成指数在均线下方且 MA20
# 斜率为负时停止新开仓（只用 t 及之前数据，无前视）。
#
# 判定：同种子配对差 vs 已有的 T069_s{seed}_k40_{win} 基线（K=40 筛选口径），n=4，
# 熊市为主判、MDE 1.64 z；牛市 MDE 1.13 z。零训练，8 次回测。
# 预期非对称：择时在熊市应该救回撤，在牛市应该因空仓错过上涨而受损——
# 所以**两窗口都要看**，只在熊市改善且牛市损失小于熊市增益时才算过。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# 等前一轮（T074）跑完再开，避免 4 路并行把内存打爆（08-08 事故）。
for i in $(seq 1 240); do
  if [ -f diagnose_output/.e6_lambda.lock ]; then
    old=$(cat diagnose_output/.e6_lambda.lock 2>/dev/null)
    if ps -p "$old" >/dev/null 2>&1; then sleep 30; continue; fi
  fi
  break
done

LOCK="diagnose_output/.regime_filter.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

name_of () { case "$1" in 42) echo "T045" ;; *) echo "T068_seed$1" ;; esac; }
dir_of  () { case "$1" in 42) echo "models/nam_gate/T045" ;; *) echo "models/nam_gate/T068_seed$1" ;; esac; }

bt_one () {
  SEED="$1"; WIN="$2"
  if [ "$WIN" = "bull" ]; then S=2024-08-05; E=2026-08-05; else S=2022-09-05; E=2024-08-05; fi
  MODEL=$(dir_of "$SEED"); NAME=$(name_of "$SEED")
  TAG="T075_s${SEED}_k40_trend_${WIN}"
  # run_backtest.py 会把 regime-filter 写进目录名（风控层参数必须进目录名，见 T040 起的约定）
  OUT=$(printf 'backtest_result/%s/nam/conf0_%s_minp1_nost_mp40_regime-trend_%s_to_%s' "$NAME" "$TAG" "$S" "$E")
  TRADES="${OUT}/backtest_trades.csv"
  if [ ! -f "$TRADES" ]; then
    echo "[BT ] $TAG"
    "$PY" scripts/run_backtest.py \
      --model "$MODEL" --start "$S" --end "$E" \
      --min-confidence 0 --risk-min-price 1 --exclude-st \
      --max-positions 40 --regime-filter trend --tag "$TAG" \
      > "diagnose_output/${TAG}_backtest.log" 2>&1
  fi
  if [ ! -f "$TRADES" ]; then
    actual=$(ls -d backtest_result/${NAME}/nam/*${TAG}* 2>/dev/null | head -1)
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

for SEED in 42 11 23 37; do
  bt_one "$SEED" bear & P1=$!
  bt_one "$SEED" bull & P2=$!
  wait $P1 $P2
  echo "--- seed=$SEED 完成 ---"
done
echo "=== T075 REGIME-FILTER DONE ==="
