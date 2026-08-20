#!/usr/bin/env bash
# T126：给 T125（数据驱动簇 + 族输出归一化）补两窗回测留档。
#
# 为什么补：IC 侧已三门全否（vs T115 跌日 ΔIC −0.104 = 6.5× 跌日 MDE），
#   本不必再花机时；用户要求留档，且 2×2 的最后一格有完整回测记录才好封轴。
# tag 沿用 T120_ 前缀，使 scripts/exp/collect_backtest_matrix.py --tag T120
#   能把它并进同一张表与 T113/T115/T118/T119 直接横比。
# 判读一律先看日频配对（scripts/exp/diag_backtest_daily_paired.py），
#   终值收益在 n=1 种子下分辨率 80~100pp（[[backtest-mde-is-80pp]]）。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
W1_S="2022-09-05"; W1_E="2024-08-05"
W2_S="2024-08-05"; W2_E="2026-08-05"
LOG=diagnose_output/T126_driver.log
A="T125_clust_gnorm_s42"
M="models/nam_gate/${A}/nam_gate_factor_model.pkl"
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
[ -f "$M" ] || { say "缺存档 $M"; exit 2; }
for W in "$W1_S $W1_E" "$W2_S $W2_E"; do
  set -- $W
  MARK="diagnose_output/.bt126_${A}_${1}.done"
  OUT="diagnose_output/T126_bt_${A}_${1}.log"
  [ -f "$MARK" ] && { say "跳过 $1（已完成）"; continue; }
  say "回测 $A  $1 → $2"
  "$PY" -u scripts/run_backtest.py --start "$1" --end "$2" \
      --tag "T120_${A}" --model "$M" > "$OUT" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "失败 $1 rc=$rc"; tail -25 "$OUT"; else touch "$MARK"; fi
done
say "完成"
