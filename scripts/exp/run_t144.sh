#!/bin/bash
# T144 验证：专家输出层零初始化（确定性起点）能否压低 σ_seed
# 同 T143 base 配置（factors_cache 247列、800股×13年、disable-gate），
# 仅加 --expert-zero-output。4 种子 → 熊市窗回测 + 随机零假设。
# 判定：σ_seed(z) vs T143 base 的 1.04（若显著下降 → 确定性起点有效）
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd /g/ai_proj/quantified_decision

COMMON="--stocks 800 --years 13 --end 2022-09-05 --disable-gate --target returns
  --y-scale 2.0 --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline
  --folds 0.8:1.0 --seeds 42,11,23,37 --allow-degenerate-downside-risk
  --expert-zero-output"

echo "===== [$(date +%H:%M:%S)] T144 零输出初始化训练开始 ====="
$PY scripts/train_nam_model.py $COMMON \
  --cache-dir "database/system_data/factors_cache" \
  --save-model-dir "models/nam_gate/T144_zero" \
  --plot-dir "diagnose_output/t144_zero" \
  --output "diagnose_output/t144_zero" \
  > diagnose_output/t144_zero_train.log 2>&1
echo "===== [$(date +%H:%M:%S)] T144 训练完成，熊市回测开始 ====="

$PY scripts/exp/run_backtest_batch.py \
  --models models/nam_gate/T144_zero_s42 models/nam_gate/T144_zero_s11 models/nam_gate/T144_zero_s23 models/nam_gate/T144_zero_s37 \
  --start 2022-09-05 --end 2024-08-05 \
  --max-positions 20 --n-sims 2000 --tag t144_bear \
  --cache-dir database/system_data/factors_cache \
  > diagnose_output/t144_eval_bear.log 2>&1

echo "===== [$(date +%H:%M:%S)] T144 全部完成 ====="
