#!/bin/bash
# T132 因子去冗余训练（phase1-factor-eng）
# 配置与 T045 完全一致，仅特征子集不同：shortlist 33 因子（drop 195 列）
# 4 种子 × ~8min/种子 ≈ 32min。日志 diagnose_output/t132_driver.log
set -e
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd /g/ai_proj/quantified_decision

for S in 42 11 23 37; do
  echo "===== [$(date +%H:%M:%S)] T132 seed=$S 开始 ====="
  $PY scripts/train_nam_model.py \
    --stocks 800 --years 13 --end 2022-09-05 --disable-gate \
    --target returns --y-scale 2.0 --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline \
    --folds "0.8:1.0" --seed "$S" \
    --drop-features-file factor_eng/drop_features_t132.txt \
    --save-model-dir "models/nam_gate/T132_shortlist_s$S" \
    --plot-dir "diagnose_output/t132_s$S" \
    --output "diagnose_output/t132_s$S"
  echo "===== [$(date +%H:%M:%S)] T132 seed=$S 完成 ====="
done
echo "===== [$(date +%H:%M:%S)] T132 全部完成 ====="
