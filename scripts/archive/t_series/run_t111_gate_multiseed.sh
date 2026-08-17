#!/usr/bin/env bash
# T111：补完 T108 门控臂的 s23/s37，**单进程双种子 + fp16 驻留**，然后回测。
#
# 已具备的前提：
#   · T109：fp16 驻留（计算仍 fp32）修掉显存超配分页，383.4→30.7s/epoch（12.5x）。
#   · T110：等价性 PASS —— s11 fp32 holdout 0.09818 vs fp16 0.09733（Δ −0.00085，
#     阈值 0.002），且**早停在同一 epoch 42**。故 s23/s37 用 fp16 可直接进 T108 判决。
#   · T110 新增 --seeds：两个种子复用同一份数据集，再省一次 11.5 分钟的构建。
#
# 判据仍是 T108 预注册的那套（同种子配对回测 Δ超额，n=4）：
#   ① 两窗都不输超过窗内 MDE；② 合计 Δ > 合并 MDE ⇒ 门控进生产；③ ① 不过 ⇒ 关轴。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"

echo "[TRAIN] s23+s37 单进程 fp16  $(date '+%H:%M:%S')"
"$PY" -u scripts/exp/exp_nam_gate.py \
  --stocks 6000 --years 13 --end 2022-09-05 --target returns --skip-baseline \
  --allow-degenerate-downside-risk --cache-dir "$CACHE" \
  --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
  --epochs 60 --min-epochs 20 --y-scale 2 --expert-hidden 16 --lr 2e-3 \
  --gate-mode softmax --lambda-lb 0.01 --store-dtype fp16 --seeds 23,37 \
  --save-model-dir "models/nam_gate/T106_lb001" \
  --plot-dir "diagnose_output/nam_gate_T106_lb001" \
  --output "diagnose_output/T106_lb001.json" \
  > diagnose_output/T111_train_s23_s37.log 2>&1
echo "[$([ $? -eq 0 ] && echo OK || echo FAIL)] 训练结束  $(date '+%H:%M:%S')"

for S in 23 37; do
  M="models/nam_gate/T106_lb001_s${S}/nam_gate_factor_model.pkl"
  [ -f "$M" ] || { echo "[SKIP] 缺存档 s$S"; continue; }
  for W in "2022-09-05 2024-08-05" "2024-08-05 2026-08-05"; do
    set -- $W
    mk="diagnose_output/.bt108_s${S}_${1}.done"
    [ -f "$mk" ] && continue
    echo "[$(date '+%H:%M:%S')] 回测 gate s$S $1"
    "$PY" -u scripts/run_backtest.py --start "$1" --end "$2" --cache-dir "$CACHE" \
        --tag "T108_gate_s${S}" --model "$M" \
        > "diagnose_output/T108_bt_gate_s${S}_${1}.log" 2>&1
    [ $? -eq 0 ] && touch "$mk"
  done
done
echo "[DONE] $(date '+%H:%M:%S')"
