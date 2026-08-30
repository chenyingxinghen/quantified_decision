#!/bin/bash
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"

# yscale=2 已完成 (00:33, 92min)
# 以下仅跑 yscale=4，与 yscale=2 同口径（不显式指定 chunk-days，走默认 chunk_days=1）
# （chunk=4 在 329 列主机驻留路径反而更慢：CPU 组装成本放大 4x 而步数只减 4x）

echo "===== [$(date +%H:%M:%S)] T147 yscale=4 seed42 开始训练 ====="
$PY scripts/train_nam_model.py \
  --stocks 5480 --years 13 --end 2022-09-05 \
  --disable-gate --target returns --y-scale 4 \
  --expert-hidden 16 --lr 2e-3 --drop-groups forecast \
  --select-holdout 0.4 --store-dtype auto \
  --seed 42 --skip-baseline \
  --save-model-dir models/nam_gate/T147_yscale4_s42 \
  --output diagnose_output/t147_yscale4/results.json \
  > diagnose_output/t147_yscale4_train.log 2>&1
echo "===== [$(date +%H:%M:%S)] T147 yscale=4 训练完成 ====="
echo "ALL DONE"