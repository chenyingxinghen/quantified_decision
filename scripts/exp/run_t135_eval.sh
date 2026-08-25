#!/bin/bash
# T135 评估：8 模型（gate/base × 4 种子）× 牛熊双窗 批量回测 + 随机零假设
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd /g/ai_proj/quantified_decision
MODELS="models/nam_gate/T135_full_gate_s42 models/nam_gate/T135_full_gate_s11 models/nam_gate/T135_full_gate_s23 models/nam_gate/T135_full_gate_s37 models/nam_gate/T135_full_base_s42 models/nam_gate/T135_full_base_s11 models/nam_gate/T135_full_base_s23 models/nam_gate/T135_full_base_s37"

echo "===== [$(date +%H:%M:%S)] 熊市窗 2022-09-05→2024-08-05 ====="
$PY scripts/exp/run_backtest_batch.py \
  --models $MODELS --start 2022-09-05 --end 2024-08-05 \
  --max-positions 20 --n-sims 2000 --tag t135_bear \
  > diagnose_output/t135_eval_bear.log 2>&1

echo "===== [$(date +%H:%M:%S)] 牛市窗 2024-08-05→2026-08-05 ====="
$PY scripts/exp/run_backtest_batch.py \
  --models $MODELS --start 2024-08-05 --end 2026-08-05 \
  --max-positions 20 --n-sims 2000 --tag t135_bull \
  > diagnose_output/t135_eval_bull.log 2>&1

echo "===== [$(date +%H:%M:%S)] T135 评估全部完成 ====="
