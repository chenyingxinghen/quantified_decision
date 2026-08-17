#!/usr/bin/env bash
# T070 / E3：标签残差化训练（beta_size）
#
# 与 T045 完全同配置，唯一变量 = --label-residualize beta_size。
# 多种子成对：与 σ_seed 池中的 {42,11,23,37} 一一配对，
# 判定用「逐种子随机分位中位数」对比同种子的 none 基线，禁止用 Val IC 判定
# （残差化删掉的正是最可预测成分，IC 必然下降）。
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"

MODE="beta_size"
for SEED in 42 11 23 37; do
  OUT="models/nam_gate/T070_e3_${MODE}_s${SEED}"
  LOG="diagnose_output/T070_e3_${MODE}_s${SEED}_train.log"
  if [ -f "$OUT/nam_gate_model.pkl" ] || [ -f "$OUT/model.pkl" ]; then
    echo "[SKIP] $OUT 已存在"
    continue
  fi
  echo "[TRAIN] seed=$SEED mode=$MODE -> $OUT"
  "$PY" scripts/exp/exp_nam_gate.py \
    --stocks 800 --years 13 --end 2022-09-05 \
    --disable-gate --target returns \
    --label-residualize "$MODE" \
    --y-scale 2 --lambda-lb 0 \
    --epochs 60 --min-epochs 20 --skip-baseline \
    --folds "0.8:1.0" --seed "$SEED" \
    --allow-degenerate-downside-risk \
    --save-model-dir "$OUT" \
    --output "diagnose_output/T070_e3_${MODE}_s${SEED}.json" \
    > "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] seed=$SEED rc=$rc  见 $LOG"
    tail -20 "$LOG"
  else
    echo "[OK  ] seed=$SEED"
  fi
done
echo "[DONE] E3 训练完成"
