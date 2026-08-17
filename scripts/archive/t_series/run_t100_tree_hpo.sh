#!/usr/bin/env bash
# T100：树模型超参搜索 —— **主判据是逐日 top-20 超额，不是整截面 IC**。
#
# **为什么要做**：树的现行配置来自冻结基线（fbec76d），当时审的是泄露/过拟合，
# 从来没有系统扫过超参。而且修好的时序 select-holdout 尺子此前只装在 NAM 上，
# 树这边的早停仍是「选型与报告同一集合」。
#
# **为什么主判据换成 top-20 超额**（2026-08-16 用户提出）：
#   lambdarank 的 ndcg 有头部截断与特有优化，整截面 Rank IC 会**低估**它 ——
#   一个只把前 20 名排对、中后段乱排的模型，IC 平平但正是生产要的。
#   实测过 lgb 的选型偏差是**负**的（−0.003）：ndcg@20 早停在 109~161 棵树，
#   远没到 IC 最优点，说明两个指标的最优点确实不在一处。
#   但「用真实回测验证头部能力」功率不够：n=4 配对回测 MDE 实测 79.76pp
#   （[[backtest-mde-is-80pp]]），几 pp 的头部差异测不出来。
#   **逐日 top-20 超额是同一件事的高功率版本**：同样直接度量「买进的那 20 只赚不赚」，
#   但样本是数百个交易日而不是 4 个终值。所以 HPO 用它，回测只做最后的可交易性验收。
#
# **判据**：4/4 种子同向 + Δ 中位过配对 MDE，主看 top20_excess_holdout，
#   副看 rank_ic_holdout。两者背离时以 top20 为准，但必须在报告里写明背离
#   —— 那正是「头部强但整截面平」的证据，本身是结论。
#
# 臂设计（每次只动一个轴，base = 现行冻结配置）：
#   base    现行 ModelConfig 参数
#   lr03    learning_rate 0.03（默认更小/更大视配置而定，见 --params 打印）
#   deep    max_depth +2 / num_leaves 翻倍：头部需要更细的交互
#   nd40    lgb 专属，ndcg 截断 @40（当前 @20）：截断放宽能否让梯度更稳
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
COMMON="--stocks 800 --years 13 --end 2022-09-05 --drop-groups forecast \
--cache-dir $CACHE --folds 0.6:0.8,0.7:0.9,0.8:1.0 --allow-degenerate-downside-risk \
--select-holdout 0.4 --topk 20"

run_arm () {  # run_arm <模型> <标签> <JSON超参覆盖>
  local model="$1" tag="$2" pj="$3"
  for S in 42 11 23 37; do
    local J="diagnose_output/T100_${model:0:3}_${tag}_s${S}.json"
    [ -f "$J" ] && { echo "[SKIP] $J"; continue; }
    echo "[TRAIN] $model/$tag seed=$S  $(date '+%H:%M:%S')"
    "$PY" -u scripts/exp/exp_tree_vs_nam.py --model "$model" $COMMON --seed "$S" \
      ${pj:+--params "$pj"} --output "$J" \
      > "diagnose_output/T100_${model:0:3}_${tag}_s${S}.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] $model/$tag s$S rc=$rc"
                       tail -12 "diagnose_output/T100_${model:0:3}_${tag}_s${S}.log"; }
  done
}

run_arm lightgbm base ''
run_arm lightgbm lr03 '{"learning_rate":0.03}'
run_arm lightgbm deep '{"num_leaves":63,"max_depth":8}'
run_arm lightgbm nd40 '{"eval_at":[40]}'
run_arm xgboost  base ''
run_arm xgboost  lr03 '{"learning_rate":0.03}'
run_arm xgboost  deep '{"max_depth":8}'

echo "[DONE] $(date '+%H:%M:%S')"
"$PY" -u scripts/archive/t_series/analyze_tree_hpo.py
