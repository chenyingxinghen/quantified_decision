#!/usr/bin/env bash
# T082 / E12：IC 可迁移性诊断（零训练，一次数据加载评估 8 个已存模型）
#
# 基线模型（T069 回测用的就是这批）：seed42=T045，其余=T068_seed{11,23,37}
# 多期标签模型：T081_mh_s{42,11,23,37}
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e12.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

"$PY" scripts/exp/diag_ic_by_regime.py \
  --model base_s42=models/nam_gate/T045 \
  --model base_s11=models/nam_gate/T068_seed11 \
  --model base_s23=models/nam_gate/T068_seed23 \
  --model base_s37=models/nam_gate/T068_seed37 \
  --model mh_s42=models/nam_gate/T081_mh_s42 \
  --model mh_s11=models/nam_gate/T081_mh_s11 \
  --model mh_s23=models/nam_gate/T081_mh_s23 \
  --model mh_s37=models/nam_gate/T081_mh_s37 \
  --stocks 800 --years 13 --end 2026-08-05 --topk 40 \
  --out diagnose_output/T082_ic_by_regime.json
echo "=== T082 E12 DONE ==="
