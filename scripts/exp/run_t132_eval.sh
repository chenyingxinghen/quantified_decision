#!/bin/bash
# T132 评估：4 种子 × 熊市窗 批量回测 + 随机零假设（T069 协议）
# 依赖 run_backtest_batch.py（R4 单进程复用因子面板，约 6min/4 模型 + 随机零假设）
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd /g/ai_proj/quantified_decision

$PY scripts/exp/run_backtest_batch.py \
  --models \
    models/nam_gate/T132_shortlist_s42 \
    models/nam_gate/T132_shortlist_s11 \
    models/nam_gate/T132_shortlist_s23 \
    models/nam_gate/T132_shortlist_s37 \
  --start 2022-09-05 --end 2024-08-05 \
  --max-positions 20 --n-sims 2000 --tag t132 \
  > diagnose_output/t132_eval_bear.log 2>&1
echo "EXIT=$?"
