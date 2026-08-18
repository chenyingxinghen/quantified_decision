#!/usr/bin/env bash
# T114：因子瘦身臂（219 → 173 列）× 4 种子，配对基线 = T113（同 fp16 同进程通道）。
#
# 杀名单 scripts/exp/t114_drop_features.txt（46 列，三来源可审计）：
#   [dead] 38 列：T098 4 种子专家输出 std<0.005 交集 —— 模型自己后验学出的冗余
#   [dup]   2 列：sqrt_atr_14 / sqrt_natr_28，当日 rank 后与 base 完全同列
#   [cov]   7 列：填充原子>50%（peg 族增长≤0 无定义、MBRevenue 79% 缺失）
#
# 预注册判据（瘦身不是晋级轴，是"无害化"轴）：
#   门A 不劣化：配对 Δ(T114−T113) holdout IC，若 0/4 或 4/4 为负且 |Δ中位|>MDE
#              → 判劣化，瘦身回退，报告哪个来源可疑（第一嫌疑：peg 族 [cov]，
#              其中 dupontAssetTurn_div_peg 在 4 种子平均重要性 top30 内）。
#   门B 跌日不塌：跌日 ΔIC 不得 4/4 为负（regime-stratified 否决门）。
#   过门后的收益按显存与归因清晰度记账，不按 IC 记账。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"

"$PY" -u scripts/exp/exp_nam_gate.py \
  --stocks 6000 --years 13 --end 2022-09-05 --disable-gate --target returns \
  --skip-baseline --allow-degenerate-downside-risk --cache-dir "$CACHE" \
  --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
  --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
  --y-scale 2 --expert-hidden 16 --lr 2e-3 \
  --store-dtype fp16 --seeds 42,11,23,37 \
  --drop-features-file scripts/exp/t114_drop_features.txt \
  --save-model-dir "models/nam_gate/T114_slim" \
  --plot-dir "diagnose_output/nam_gate_T114" \
  --output "diagnose_output/T114_slim.json"
rc=$?
echo "[T114] exit=$rc  $(date '+%H:%M:%S')"
exit $rc
