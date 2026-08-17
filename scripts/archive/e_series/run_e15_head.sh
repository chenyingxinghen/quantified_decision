#!/usr/bin/env bash
# T085 / E15：打分的**极端头部**还有没有分辨力（零训练）
#
# 线索：基线训练输出的 head_pct 显示 top1% 的实现收益分位（0.575~0.630）并不优于
# top20%（0.608~0.615），4 个种子里 3 个更差。而生产是 ~5000 选 40 ＝ top 0.8%，
# 正好落在这一档。若确认头部平坦 ⇒ 该动的是**损失函数的头部权重**（ListNet 现在
# 优化的是全截面序），而不是再去找特征（E8 已证特征侧到顶）。
#
# 用全市场随机 800 只评估：板块构成接近部署面板（见 T084）。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e15.lock"
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
  --stocks 800 --years 13 --end 2026-08-05 --topk 40 \
  --code-mode random --head-profile \
  --out diagnose_output/T085_head_profile.json
echo "=== T085 E15 DONE ==="
