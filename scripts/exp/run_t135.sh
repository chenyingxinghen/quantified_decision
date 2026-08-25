#!/bin/bash
# T135 门控重验（用户指令：门控有作用→全量数据+不冻结expert+多种子+牛熊分段）
# 实验设计（配对，T069 协议）：
#   gate 组：dual_scalar + mkt_pc1/pc2 + include-mkt + gate_wd 0.1，**不冻结 expert**
#            （去掉 scheme B 的 --init-experts-from/--freeze-experts，gate+expert 端到端）
#   base 组：--disable-gate（同数据同种子对照）
#   数据：全量 5480 只 × 8 年（2014-08→2022-09，内存安全 ~15GB）
#   种子：42,11,23,37（--seeds 单进程复用数据集）
# 后续：牛熊双窗回测 + 随机零假设（run_t135_eval.sh）
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd /g/ai_proj/quantified_decision
COMMON="--stocks 5480 --years 8 --end 2022-09-05 --target returns --y-scale 2.0
  --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline --folds 0.8:1.0
  --seeds 42,11,23,37"

echo "===== [$(date +%H:%M:%S)] T135 gate 组（不冻结端到端）开始 ====="
$PY scripts/train_nam_model.py $COMMON \
  --gate-mode dual_scalar --gate-scalar-col mkt_pc1 --gate-scalar-col2 mkt_pc2 \
  --include-mkt --gate-wd 0.1 \
  --save-model-dir "models/nam_gate/T135_full_gate" \
  --plot-dir "diagnose_output/t135_gate" \
  --output "diagnose_output/t135_gate" \
  > diagnose_output/t135_gate_train.log 2>&1
echo "===== [$(date +%H:%M:%S)] T135 gate 组完成 ====="

echo "===== [$(date +%H:%M:%S)] T135 base 组（disable-gate）开始 ====="
$PY scripts/train_nam_model.py $COMMON \
  --disable-gate \
  --save-model-dir "models/nam_gate/T135_full_base" \
  --plot-dir "diagnose_output/t135_base" \
  --output "diagnose_output/t135_base" \
  > diagnose_output/t135_base_train.log 2>&1
echo "===== [$(date +%H:%M:%S)] T135 base 组完成 ====="
