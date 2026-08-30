#!/bin/bash
# T143 配对重训：基线 factors_cache(247) vs +msens90+turn/val11 factors_cache_beta(348)
# T069 协议：同种子只换 cache-dir，4 种子(42,11,23,37)，T045 同配置(800股×13年)
# 后续：牛熊双窗真实成本回测 + 随机零假设 2000 sims（run_t143_eval.sh）
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd /g/ai_proj/quantified_decision

COMMON="--stocks 800 --years 13 --end 2022-09-05 --disable-gate --target returns
  --y-scale 2.0 --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline
  --folds 0.8:1.0 --seeds 42,11,23,37 --allow-degenerate-downside-risk"

echo "===== [$(date +%H:%M:%S)] 基线组 factors_cache 开始 ====="
$PY scripts/train_nam_model.py $COMMON \
  --cache-dir "database/system_data/factors_cache" \
  --save-model-dir "models/nam_gate/T143_base" \
  --plot-dir "diagnose_output/t143_base" \
  --output "diagnose_output/t143_base" \
  > diagnose_output/t143_base_train.log 2>&1
echo "===== [$(date +%H:%M:%S)] 基线组完成 ====="

echo "===== [$(date +%H:%M:%S)] 实验组 factors_cache_beta 开始 ====="
$PY scripts/train_nam_model.py $COMMON \
  --cache-dir "database/system_data/factors_cache_beta" \
  --save-model-dir "models/nam_gate/T143_beta" \
  --plot-dir "diagnose_output/t143_beta" \
  --output "diagnose_output/t143_beta" \
  > diagnose_output/t143_beta_train.log 2>&1
echo "===== [$(date +%H:%M:%S)] 实验组完成 ====="
