#!/usr/bin/env bash
# T096：修好评估器后的 NAM 调参。
#
# **为什么调参必须等评估器修好**：此前报告的 rank_ic 就是 40 个 epoch 里的最大值，
# 选 checkpoint 和报告用同一个验证集，实测虚高 25~35%，且虚高幅度随种子摆动
# 0.0189~0.0260（见 [[checkpoint-selection-inflates-ic]]）。拿这种指标做 HPO，
# 选出来的是「哪组超参更容易在验证集上撞到高点」，不是「哪组更强」。
# 现在 --select-holdout 0.4 把验证折切成 前60%选型 / 后40%报告，报告段对早停不可见。
#
# **判据**：横比只看 rank_ic_holdout（无偏）。仍按 4/4 同向 + 下跌日分层，
# 但注意 holdout 段样本量只有原来的 40%，MDE 会变大 —— 所以只做**粗筛**，
# 不追 0.001 量级的差异。
#
# **为什么在 800 只股票上调**：全量 5480 只单跑约 3 小时，网格 × 4 种子跑不完。
# 800 只单跑 26 分钟，先在这个尺度上定超参，再把胜出配置放大到全量。
# 前提假设：超参的相对优劣不随股票池规模翻转 —— 这是假设，不是已验证结论。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
COMMON="--stocks 800 --years 13 --end 2022-09-05 --disable-gate --target returns \
--skip-baseline --allow-degenerate-downside-risk --cache-dir $CACHE \
--folds 0.6:0.8,0.7:0.9,0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
--epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4"

# --chunk-days 4：批前向路径（T097），训练段约 3.8x。optimizer 步频仍锁在
# --accum-days，与逐日路径语义等价 —— 同种子同折实测 rank_ic 0.07796→0.07835、
# holdout 0.06609→0.06648、早停 epoch 同为 33，差值是浮点求和顺序的量级。
# 2026-08-15 10:47 重启：逐日路径已跑完 base 全 4 种子 + ys4 s42/s11，
# 那批产物移到 diagnose_output/iter_output/T096_perday/ 留档。本次全部重跑，
# 因为混用两条路径的臂间比较会把 ±0.0004 的实现差异掺进判决。

# 臂：base=T095 冠军配置；其余每次只动一个轴
run_arm () {  # run_arm <标签> <额外参数...>
  local tag="$1"; shift
  for S in 42 11 23 37; do
    local J="diagnose_output/T096_${tag}_s${S}.json"
    [ -f "$J" ] && { echo "[SKIP] $J"; continue; }
    echo "[TRAIN] $tag seed=$S  $(date '+%H:%M:%S')"
    "$PY" -u scripts/exp/exp_nam_gate.py $COMMON --seed "$S" "$@" \
      --output "$J" > "diagnose_output/T096_${tag}_s${S}_train.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] $tag s$S rc=$rc"; tail -12 "diagnose_output/T096_${tag}_s${S}_train.log"; }
  done
}

run_arm base   --y-scale 2
run_arm ys4    --y-scale 4          # 头部权重：e²≈7.4 → e⁴≈55
run_arm eh32   --y-scale 2 --expert-hidden 32   # 单因子形状函数容量翻倍
run_arm lr1e3  --y-scale 2 --lr 1e-3            # 学习率减半（配合更长收敛）

echo "[DONE] $(date '+%H:%M:%S')"
"$PY" -u scripts/exp/analyze_holdout.py --prefix T096 --arms base,ys4,eh32,lr1e3
