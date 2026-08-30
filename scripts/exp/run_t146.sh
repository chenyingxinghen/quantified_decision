#!/bin/bash
# T146 验证：多种子打乱（multi-seed shuffle）—— 把后训练种子集成的多样性搬进训练。
# 用户设想：与其训练 4 个模型各自走一条确定轨迹再集成，不如让单模型在训练中
# 每个 epoch 轮转使用种子集 {42,11,23,37} 的独立排列，迫使参数在所有种子日序轨迹
# 下都好 → 压制日序洗牌彩票（T144 实证 σ_seed 主成分的 84%），同时保留信号。
# 单轴对照：仅加 --multi-seed-shuffle，其余完全同 T143 base（factors_cache 247 列、
# 800 股×13 年、disable-gate、4 种子）。4 种子 → 熊市窗回测 + 随机零假设。
# 判定基准（T069 配对协议，熊市主判窗、z 口径）：
#   T143 base (单种子洗牌)  σ_seed(z)=1.040  z̄=+1.247
#   T144 zero  (洗牌+零输出) σ_seed(z)=0.872  z̄=+0.505 (-60%) 不晋级
#   T145 fixed (固定日序)    σ_seed(z)=0.603  z̄=-0.153 (崩) 不晋级
#   目标：T146 σ_seed 较 T143 显著下降（逼近 init-only ~0.6），且 z 均值不被腰斩
#         （区别于 T145：每个 epoch 仍是随机排列，无时间偏置，不当成时序捷径）。
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd /g/ai_proj/quantified_decision

COMMON="--stocks 800 --years 13 --end 2022-09-05 --disable-gate --target returns
  --y-scale 2.0 --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline
  --folds 0.8:1.0 --seeds 42,11,23,37 --allow-degenerate-downside-risk
  --multi-seed-shuffle --multi-seed-shuffle-seeds 42,11,23,37"

echo "===== [$(date +%H:%M:%S)] T146 多种子打乱训练开始 ====="
$PY scripts/train_nam_model.py $COMMON \
  --cache-dir "database/system_data/factors_cache" \
  --save-model-dir "models/nam_gate/T146_msshuf" \
  --plot-dir "diagnose_output/t146_msshuf" \
  --output "diagnose_output/t146_msshuf" \
  > diagnose_output/t146_msshuf_train.log 2>&1
echo "===== [$(date +%H:%M:%S)] T146 训练完成，熊市回测开始 ====="

$PY scripts/exp/run_backtest_batch.py \
  --models models/nam_gate/T146_msshuf_s42 models/nam_gate/T146_msshuf_s11 models/nam_gate/T146_msshuf_s23 models/nam_gate/T146_msshuf_s37 \
  --start 2022-09-05 --end 2024-08-05 \
  --max-positions 20 --n-sims 2000 --tag t146_bear \
  --cache-dir database/system_data/factors_cache \
  > diagnose_output/t146_eval_bear.log 2>&1

echo "===== [$(date +%H:%M:%S)] T146 全部完成 ====="
