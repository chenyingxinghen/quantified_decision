#!/usr/bin/env bash
# T108：门控臂（全量池 + lambda_lb 0.01）扩到 4 种子 —— IC 与回测方向相反，必须判清。
#
# 单种子 42 的读数：
#   IC holdout 0.09719 vs 纯加性 0.11114（Δ −0.01395 = 反向 2.3× 配对 MDE）
#   回测熊市 +29.53pp vs 同种子 +0.51pp（Δ +29.02pp）
#   回测牛市 +18.46pp vs 同种子 +18.98pp（Δ −0.52pp，基本打平）
#   两窗都超过 4 个纯加性种子里的 3 个，各 +0.63σ / +0.69σ。
#
# 这是台账里**第一个 IC 与回测方向相反且回测两窗都不输**的臂。
# 之前所有「回测好看」的臂要么靠 β（blend，T105）、要么一窗赢一窗输（E6 门控），
# 这个都不是：β 1.182/0.871 与纯加性同种子的 1.199/0.833 几乎一样，胜率两窗都高。
#
# **判据（预注册）**：同种子配对 Δ超额，n=4。
#   ① 两窗都不输超过窗内 MDE（无 regime 债）；
#   ② 合计 Δ > 合并 MDE ⇒ 门控进生产（尽管 IC 更低 —— 那说明整截面 IC 不是正确的
#      载体判据，本身是重要结论，须单独记录）；
#   ③ 若 ① 不过 ⇒ s42 是运气，门控轴永久关闭。
# ⚠ IC 侧已明确劣化，所以这一轮**只能靠回测晋级**。若晋级，必须在台账里写明
#   「用回测推翻 IC 判定」这个先例，以及它对 [[alpha-not-rules-ic-first-protocol]] 的修正。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
COMMON="--stocks 6000 --years 13 --end 2022-09-05 --target returns --skip-baseline \
--allow-degenerate-downside-risk --cache-dir $CACHE \
--folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
--epochs 60 --min-epochs 20 --y-scale 2 --expert-hidden 16 --lr 2e-3 \
--gate-mode softmax --lambda-lb 0.01"

for S in 11 23 37; do
  J="diagnose_output/T106_lb001_s${S}.json"
  if [ ! -f "$J" ]; then
    echo "[TRAIN] gate s$S  $(date '+%H:%M:%S')"
    "$PY" -u scripts/exp/exp_nam_gate.py $COMMON --seed "$S" \
      --save-model-dir "models/nam_gate/T106_lb001_s${S}" \
      --plot-dir "diagnose_output/nam_gate_T106_lb001_s${S}" \
      --output "$J" > "diagnose_output/T106_lb001_s${S}.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] s$S rc=$rc"; tail -12 "diagnose_output/T106_lb001_s${S}.log"; continue; }
    echo "[OK] train s$S  $(date '+%H:%M:%S')"
  fi
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
