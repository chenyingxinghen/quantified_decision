#!/bin/bash
# T143 评估：8 模型（base/beta × 4 种子）× 牛熊双窗 批量回测 + 随机零假设
# 配对协议：同种子 base vs beta 只差缓存列；熊市主判窗、z 口径(T069)
# 注意：2026-08-22 起 resolve_model_cache 硬编码返回共享 factors_cache，
# 独立 factors_cache_beta 必须显式 --cache-dir；base/beta 分两批跑。
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd /g/ai_proj/quantified_decision

BASE="models/nam_gate/T143_base_s42 models/nam_gate/T143_base_s11 models/nam_gate/T143_base_s23 models/nam_gate/T143_base_s37"
BETA="models/nam_gate/T143_beta_s42 models/nam_gate/T143_beta_s11 models/nam_gate/T143_beta_s23 models/nam_gate/T143_beta_s37"

echo "===== [$(date +%H:%M:%S)] 熊市窗 base 4 模型 ====="
$PY scripts/exp/run_backtest_batch.py \
  --models $BASE --start 2022-09-05 --end 2024-08-05 \
  --max-positions 20 --n-sims 2000 --tag t143_bear_base \
  --cache-dir database/system_data/factors_cache \
  > diagnose_output/t143_bear_base.log 2>&1

echo "===== [$(date +%H:%M:%S)] 熊市窗 beta 4 模型 ====="
$PY scripts/exp/run_backtest_batch.py \
  --models $BETA --start 2022-09-05 --end 2024-08-05 \
  --max-positions 20 --n-sims 2000 --tag t143_bear_beta \
  --cache-dir database/system_data/factors_cache_beta \
  > diagnose_output/t143_bear_beta.log 2>&1

echo "===== [$(date +%H:%M:%S)] 牛市窗 base 4 模型 ====="
$PY scripts/exp/run_backtest_batch.py \
  --models $BASE --start 2024-08-05 --end 2026-08-05 \
  --max-positions 20 --n-sims 2000 --tag t143_bull_base \
  --cache-dir database/system_data/factors_cache \
  > diagnose_output/t143_bull_base.log 2>&1

echo "===== [$(date +%H:%M:%S)] 牛市窗 beta 4 模型 ====="
$PY scripts/exp/run_backtest_batch.py \
  --models $BETA --start 2024-08-05 --end 2026-08-05 \
  --max-positions 20 --n-sims 2000 --tag t143_bull_beta \
  --cache-dir database/system_data/factors_cache_beta \
  > diagnose_output/t143_bull_beta.log 2>&1

echo "===== [$(date +%H:%M:%S)] T143 评估全部完成 ====="
