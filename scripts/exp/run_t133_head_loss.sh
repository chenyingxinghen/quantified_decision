#!/usr/bin/env bash
# T133（②）：**头部集中的损失** —— T130 之后唯一有理论支撑的新轴。
#
# 立论
#   T130 量到两件事：
#     ① 头部（前 200 候选池内重排取前 20）的**诚实折外天花板**是 top20 超额 +0.0176，
#        生产件只拿到 +0.0071 —— **2.5 倍空间**，而且这个空间是"重排族权重"就能吃到的。
#     ② 全截面 rank IC **看不见**头部改动（同一个重排在全截面上只有 +0.0011 / t=2.13，
#        persist 类甚至转负）。见 [[head-ruler-is-blind-in-full-cross-section-ic]]。
#   而现在的训练配方**两头都对着全截面**：ListNet 在全截面上算损失，早停用全截面 rank IC
#   选 checkpoint。**我们在优化一个不交易的目标。**
#
#   `--y-scale` 正好是这条轴的旋钮：ListNet 的目标是 `softmax(y * y_scale)`，
#   y_scale 越大，目标质量越集中在当日排名的**头部**。冠军 y2 是 [[nam-hpo-axes-closed]]
#   扫出来的 —— **但那轮是用全截面 IC 判的，也就是那把瞎尺子**。所以这不是重开已关的轴，
#   是**用能看见头部的尺子重判一条以前判错过的轴**。
#
# ⚠ 必须诚实说明的先验
#   y_scale 变大 ⇒ 有效样本量变小（softmax 目标趋近 one-hot，每天只有几只票贡献梯度），
#   所以既可能"对准了头部"也可能"只是噪声更大"。这就是为什么必须四种子 + 跌日否决门。
#
# ── 预注册判定（出结果前写死）───────────────────────────────────────────────
#   主判据（头部尺子，T130 口径）：逐日 top-20 超额，候选池 = 前 200，
#     四种子配对 vs y2 基线（= T115_idxrel_s*，同窗同超参，只差 y_scale）：
#       晋级需 **4/4 为正** 且 **Δ 中位 ≥ +0.002**（头部超额的量级比 IC 小一个数量级，
#       0.005 那条线是给全截面 IC 定的，直接搬过来不合口径；0.002 约等于 T130 里
#       `global` 臂 +0.0042 的一半，取"能被现有工具稳定分辨"的量级）。
#   否决门（优先于主判据）：**跌日** Δ 必须 ≥3/4 为正。
#     T130 里 `global`/`ifelse` 正是头部赢、跌日 2/4 而被拒的，同一把尺子同一条门。
#   副判据（记录用，不参与晋级）：全截面 holdout rank IC。
#     **预期它会掉** —— 若它掉而头部涨，那正是"两把尺子会给相反结论"的又一个例证；
#     若两把尺子同时涨，反而要怀疑是不是种子运气，回去看四种子离散度。
#
#   ⚠ 判定脚本：scripts/exp/diag_seed_ensemble.py 已经同时报两把尺子 + 涨跌日分层，
#     直接拿它对 {T115_idxrel_s*, T133_yscale4_s*, T133_yscale8_s*} 各跑一遍即可。
#
# 窗口必须沿用 --years 13 --end 2022-09-05：要与 T115 配对，换窗就换考卷
#   （[[holdout-ic-not-comparable-across-windows]]）。该窗 8.98M 样本 / fp16 3.75 GiB，
#   走 cuda 驻留，不需要 --store-device host。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-18-idxrel"

rc_all=0
for YS in 4 8; do
  echo "===== T133 y_scale=$YS  $(date '+%H:%M:%S') ====="
  "$PY" -u scripts/exp/exp_nam_gate.py \
    --stocks 6000 --years 13 --end 2022-09-05 --disable-gate --target returns \
    --skip-baseline --allow-degenerate-downside-risk --cache-dir "$CACHE" \
    --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
    --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
    --y-scale "$YS" --expert-hidden 16 --lr 2e-3 \
    --store-dtype fp16 --seeds 42,11,23,37 \
    --save-model-dir "models/nam_gate/T133_yscale${YS}" \
    --plot-dir "diagnose_output/nam_gate_T133_y${YS}" \
    --output "diagnose_output/T133_yscale${YS}.json"
  rc=$?
  echo "[T133 y=$YS] exit=$rc  $(date '+%H:%M:%S')"
  [ $rc -ne 0 ] && rc_all=$rc
done
echo "[T133] all done exit=$rc_all  $(date '+%H:%M:%S')"
exit $rc_all
