#!/usr/bin/env bash
# T107：T106 门控存档（全量池 + lambda_lb 0.01，门控健康未塌陷）的回测。
# IC 侧已判劣化 Δ −0.01395（跌日 0.111→0.080），但用户要求回测确认 ——
# 理由正当：IC 与回测口径不同，且门控若真在做 regime 路由，收益侧可能有 IC 看不到的
# 分布差异（T103 已有先例：IC 与回测这次一致，但那不保证下次一致）。
# 对照：T099_bt_full_s42（同池同种子的纯加性 NAM）。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
M="models/nam_gate/T106_lb001_s42/nam_gate_factor_model.pkl"
[ -f "$M" ] || { echo "缺存档 $M"; exit 1; }
for W in "2022-09-05 2024-08-05" "2024-08-05 2026-08-05"; do
  set -- $W
  out="diagnose_output/T107_bt_gate_s42_${1}.log"
  [ -f "diagnose_output/.bt107_${1}.done" ] && { echo "[SKIP] $1"; continue; }
  echo "[$(date '+%H:%M:%S')] 回测 gate  $1 → $2"
  "$PY" -u scripts/run_backtest.py --start "$1" --end "$2" --cache-dir "$CACHE" \
      --tag "T107_gate_s42" --model "$M" > "$out" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL] $1 rc=$rc"; tail -15 "$out"; } || touch "diagnose_output/.bt107_${1}.done"
done
echo "[DONE] $(date '+%H:%M:%S')"
