#!/bin/bash
# T147 评估驱动：2 模型(yscale2/4) × 2 窗口(牛/熊) 回测 + 随机零假设
# 成本 CLI 覆盖为 T048 修正后真实口径 0.0008/0.0013（strategy_config 当前仍是 0.005 旧 bug 值）
# 窗口与 T115 生产 OOS 完全一致，保证可直接对照 T115 基线。
set -u
cd /g/ai_proj/quantified_decision
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
OUT=diagnose_output/t147_eval
mkdir -p "$OUT"

MODELS=(T147_yscale2_s42 T147_yscale4_s42)
# "标签 开始 结束"
WINDOWS=(
  "bear 2022-09-05 2024-08-05"
  "bull 2024-08-05 2026-08-05"
)

run_bt() {
  local model="$1" start="$2" end="$3" tag="$4"
  local log="$OUT/bt_${model}_${tag}.log"
  echo "===== [$(date +%H:%M:%S)] backtest $model $tag ($start -> $end) =====" | tee -a "$log"
  $PY scripts/run_backtest.py \
    --model "models/nam_gate/$model" \
    --start "$start" --end "$end" \
    --max-positions 20 --min-confidence 0 \
    --buy-cost 0.0008 --sell-cost 0.0013 \
    --tag t147 \
    >> "$log" 2>&1
  echo "===== [$(date +%H:%M:%S)] backtest $model $tag done (exit $?) =====" | tee -a "$log"
}

run_null() {
  local model="$1" start="$2" end="$3" tag="$4"
  local trades="backtest_result/$model/nam/conf0_minp1_nost_mp20_${start}_to_${end}/backtest_trades.csv"
  local out="$OUT/null_${model}_${tag}.json"
  if [ ! -f "$trades" ]; then
    echo "!! trades missing for $model $tag: $trades" | tee -a "$OUT/null_${model}_${tag}.log"
    return 1
  fi
  echo "===== [$(date +%H:%M:%S)] random-null $model $tag =====" | tee -a "$OUT/null_${model}_${tag}.log"
  $PY scripts/exp/diag_random_null.py \
    --trades "$trades" --n-sims 2000 --buy-cost 0.0008 --sell-cost 0.0013 \
    --out "$out" \
    >> "$OUT/null_${model}_${tag}.log" 2>&1
  echo "===== [$(date +%H:%M:%S)] random-null $model $tag done =====" | tee -a "$OUT/null_${model}_${tag}.log"
}

for m in "${MODELS[@]}"; do
  for w in "${WINDOWS[@]}"; do
    set -- $w
    tag="$1"; start="$2"; end="$3"
    run_bt "$m" "$start" "$end" "$tag"
  done
done

for m in "${MODELS[@]}"; do
  for w in "${WINDOWS[@]}"; do
    set -- $w
    tag="$1"; start="$2"; end="$3"
    run_null "$m" "$start" "$end" "$tag"
  done
done

echo "ALL T147 EVAL DONE at $(date +%H:%M:%S)"
