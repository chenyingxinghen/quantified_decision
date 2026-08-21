#!/usr/bin/env bash
# T136 —— 生产载体终审：T115_idxrel（实盘在用）vs T122_prod9y，同卷四种子。
#
# 背景：任务 #14。T132（13y 窗）已在 T133/T132 判定里输给 T122（4/4），
# 所以候选收敛成 T122。但 **T122 从未与实盘在用的 T115 同卷横比过** ——
# 之前两者的 rank_ic_holdout 来自不同窗口，按
# [[holdout-ic-not-comparable-across-windows]] 不可直接比。
#
# 卷子：9y → 2026-08-10，train_fraction 0.8，select_holdout 0.4。
# 与 T132 vs T122 那次**完全同卷**，所以三方数字可以放进同一张表。
#
# ⚠⚠ 必须随结论一起报的偏置（这张卷子对 T122 有利）：
#   T115 训到 2020-12-18，T122 训到 2025-01-20。
#   本卷 holdout 落在 2025 年之后 ——
#     · 对 T115 是 4~5 年后的纯样本外；
#     · 对 T122 是 0~19 个月后，且**与 T122 自己的验证段有重叠**
#       ⇒ T122 的 checkpoint 选型见过本卷的一部分（选型污染，按
#          [[checkpoint-selection-inflates-ic]] 历史上值 12~13%）。
#   ⇒ **T122 赢不能直接当证据**（要扣掉这两项）；**T122 不赢则结论很硬**
#      （在对它最有利的卷子上都不赢）。
#
# 预注册判据（沿用打分层协议，跌日否决门优先）：
#   否决门：T122 的 **跌日 holdout IC** 不得劣于 T115（逐种子配对，≥3/4 不劣）。
#           否则直接不换，后面几条不看。
#   晋级 ：(a) 逐种子配对 4/4 为正
#          (b) holdout IC 中位数 Δ ≥ +0.005
#   两条都过才把 AUTO_MODEL_PATH / AUTO_NORM_STATS_PATH 换到 T122。
#   ⚠ 换载体必须两个路径**同时**改到同一个存档目录（norm_stats 与权重成对，
#     错配会静默变成 T043 那类尺度错配），改完跑 tests.test_business_logic。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"

M=""
for B in T115_idxrel T122_prod9y; do
  for SD in 42 11 23 37; do
    M="${M}${M:+,}models/nam_gate/${B}_s${SD}"
  done
done

echo "########## T136 开始 $(date '+%F %H:%M:%S') ##########"
echo "models=$M"
"$PY" -u scripts/exp/eval_models_on_window.py \
  --models "$M" --years 9 --end 2026-08-10 \
  --train-fraction 0.8 --select-holdout 0.4 \
  --out diagnose_output/T136_T115_vs_T122_same_window.json
echo "[T136] exit=$? $(date '+%F %H:%M:%S')"
echo "########## T136 结束 $(date '+%F %H:%M:%S') ##########"
