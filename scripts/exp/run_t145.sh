#!/bin/bash
# T145 验证：层次A —— 固定 epoch 内训练日序（去 rng.permutation 洗牌）
# 能否压低 σ_seed（T144 实证 σ_seed 主成分是日序洗牌，非权重初始化）。
# 单轴对照：仅加 --fixed-day-order，其余完全同 T143 base（factors_cache 247 列、
# 800 股×13 年、disable-gate、4 种子）。4 种子 → 熊市窗回测 + 随机零假设。
# 判定基准（T069 配对协议，熊市主判窗、z 口径）：
#   T143 base  σ_seed(z)=1.040
#   T144 zero  σ_seed(z)=0.872 (-16%)，但 z 均值 +1.247→+0.505 (-60%) 不晋级
#   目标：T145 σ_seed 较 T143 显著下降，且 z 均值不被腰斩（与 T144 区分）。
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd /g/ai_proj/quantified_decision

COMMON="--stocks 800 --years 13 --end 2022-09-05 --disable-gate --target returns
  --y-scale 2.0 --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline
  --folds 0.8:1.0 --seeds 42,11,23,37 --allow-degenerate-downside-risk
  --fixed-day-order"

echo "===== [$(date +%H:%M:%S)] T145 固定日序训练开始 ====="
$PY scripts/train_nam_model.py $COMMON \
  --cache-dir "database/system_data/factors_cache" \
  --save-model-dir "models/nam_gate/T145_fixedorder" \
  --plot-dir "diagnose_output/t145_fixedorder" \
  --output "diagnose_output/t145_fixedorder" \
  > diagnose_output/t145_fixedorder_train.log 2>&1
echo "===== [$(date +%H:%M:%S)] T145 训练完成，熊市回测开始 ====="

$PY scripts/exp/run_backtest_batch.py \
  --models models/nam_gate/T145_fixedorder_s42 models/nam_gate/T145_fixedorder_s11 models/nam_gate/T145_fixedorder_s23 models/nam_gate/T145_fixedorder_s37 \
  --start 2022-09-05 --end 2024-08-05 \
  --max-positions 20 --n-sims 2000 --tag t145_bear \
  --cache-dir database/system_data/factors_cache \
  > diagnose_output/t145_eval_bear.log 2>&1

echo "===== [$(date +%H:%M:%S)] T145 全部完成 ====="
