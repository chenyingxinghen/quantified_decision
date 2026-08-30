#!/bin/bash
# T133 PCA 正交压缩训练（phase1-factor-eng，用户选 B）
# 配置与 T045 一致，仅特征变换：横截面归一化后 PCA 压缩到 30 个主成分
# （不硬删因子，保留全部信息方向，压缩等价解空间 → 压 σ_seed）
# 4 种子 × ~5min/种子 ≈ 20min。日志 diagnose_output/t133_driver.log
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd /g/ai_proj/quantified_decision

for S in 42 11 23 37; do
  echo "===== [$(date +%H:%M:%S)] T133 seed=$S 开始 ====="
  $PY scripts/train_nam_model.py \
    --stocks 800 --years 13 --end 2022-09-05 --disable-gate \
    --target returns --y-scale 2.0 --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline \
    --folds "0.8:1.0" --seed "$S" \
    --pca-components 30 \
    --save-model-dir "models/nam_gate/T133_pca30_s$S" \
    --plot-dir "diagnose_output/t133_s$S" \
    --output "diagnose_output/t133_s$S"
  echo "===== [$(date +%H:%M:%S)] T133 seed=$S 完成 ====="
done
echo "===== [$(date +%H:%M:%S)] T133 全部完成 ====="
