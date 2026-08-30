#!/bin/bash
# T147 随机零假设（修正路径版）：glob 真实 trades.csv，避免 --tag 导致的目录名错位
set -u
cd /g/ai_proj/quantified_decision
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
OUT=diagnose_output/t147_eval
mkdir -p "$OUT"

shopt -s nullglob
files=(backtest_result/T147_yscale{2,4}_s42/nam/*_t147_minp1_nost_mp20_*/backtest_trades.csv)
if [ ${#files[@]} -eq 0 ]; then
  echo "!! 未找到任何 T147 trades.csv，先确认回测已完成"
  exit 1
fi
for f in "${files[@]}"; do
  model=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
  wind=$(basename "$(dirname "$f")")
  tag="${model}_${wind}"
  out="$OUT/null_${tag}.json"
  echo "===== [$(date +%H:%M:%S)] random-null $tag ====="
  $PY scripts/exp/diag_random_null.py \
    --trades "$f" --n-sims 2000 --buy-cost 0.0008 --sell-cost 0.0013 \
    --out "$out"
  echo "  -> $out"
done
echo "ALL T147 NULLS DONE at $(date +%H:%M:%S)"
