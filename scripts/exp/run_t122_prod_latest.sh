#!/usr/bin/env bash
# T122：**生产模型训至最新交易日**（配方全同 T115，只滑动时间窗）。
#
# 要解决的问题：当前生产件 T115_idxrel_s42 的专家网络只拟合到 **2020-12-18**
#   （`--folds 0.8:1.0` 把后 20% 留给早停+无偏判定；`2022-09-05` 只是验证段终点）。
#   实盘语义因此是「特征当日算、形状函数 2020 年拟合」，陈旧 5.7 年。
#
# ── 三个绕不开的取舍，全部是刻意选择，不是遗漏 ────────────────────────────────
#
# ① **失去干净的 OOS 窗口**。同一批数据不能同时做「训练」和「样本外验收」。
#    本次把 2022-09→2026-08 吃进训练+验证，代价是 T099/T120 用的那两个 OOS 窗
#    （熊 2022-09→2024-08、牛 2024-08→2026-08）**从此落在本模型的训练/验证跨度内**，
#    再对它们回测已**不是**样本外。配方已由 T095~T120 反复验过，此刻要的是时效，
#    不是再验一次配方。今后唯一诚实的评估是**向前测**（未来数据）。
#
# ② **窗口必须从 13 年缩到 9 年 —— 显存硬约束，不是偷懒**。
#    本机 RTX 4050 Laptop 只有 6141 MiB。按缓存实测样本量估算 fp16 224 列驻留：
#      13y(2013-08→2026-08) 12.11M 样本 → 5.05 GiB + 非特征开销 1.87 GiB = 6.92 GiB ✗
#       9y(2017-08→2026-08)  9.58M 样本 → 4.00 GiB + 1.87 GiB = 5.87 GiB ✓
#    尺子已用 T109 校准（旧窗估 3.75 GiB vs 实测 3.78 GiB，实测总占用 5754 MiB）。
#    超配**不会报 OOM**，会被 WDDM 静默页到内存 → T109 实测慢 **12.5x**（5h→60h）。
#    好消息：样本量与旧配方基本持平（9.58M vs 8.98M，近年上市公司更多补上了年份损失），
#    训练动力学可比，早停轮次预期仍在 30~60。
#
# ③ **早停的选型段变成纯牛市**。9y 窗的后 20% ≈ 2024-05→2026-08，基准 +95.64%。
#    旧配方的选型段是 2020-12→2022-09（混合偏熊）。按台账反复出现的
#    「涨日赢/跌日输」跷跷板，在纯牛市段上选 checkpoint 有偏向高 β 解的风险。
#    **无法回避**（最新数据就是牛市），只能事后查：产出后必看 holdout 的
#    **跌日 rank IC**，若显著低于 T115 的 0.13497，说明选型段的 regime 偏置真的进来了。
#
# ── 与 T115 的差异清单（只有这三项）──────────────────────────────────────────
#   --years 13 → 9         （显存，见②）
#   --end 2022-09-05 → 2026-08-10   （缓存实际覆盖到 2026-08-10）
#   去掉 --allow-degenerate-downside-risk
#     （实测当前缓存 downside_risk 非零率 98.81%、std 0.0092，健康；
#      新生产模型不该带 legacy 逃生阀，让守卫真的把关）
#   其余全同：--disable-gate --lambda-lb 0 --drop-groups forecast --select-holdout 0.4
#            --epochs 60 --min-epochs 20 --chunk-days 4
#            --y-scale 2 --expert-hidden 16 --lr 2e-3 --store-dtype fp16
#            --seeds 42,11,23,37
#
# ── 产出后的检查单（不做完不许接生产）──────────────────────────────────────
#   1. holdout rank IC 与 T115 的 0.12012 同量级（不同窗口不能配对判定，只看量级）
#   2. **跌日 IC** 不显著低于 0.13497（③ 的偏置检查）
#   3. 种子间 σ 不显著大于 T115 的 0.00302
#   4. 存档里 factor_cache_manifest.json / feature_names.json / norm_stats.pkl 齐全
#   5. pytest tests/test_business_logic.py -k ArtifactAndCache 全绿
#   6. 改 config/automation_config.py 的 AUTO_MODEL_PATH + AUTO_NORM_STATS_PATH（同批产出）
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-18-idxrel"
END="2026-08-10"

[ -f "$CACHE/factor_cache_manifest.json" ] || {
  echo "[ABORT] 缺缓存清单 $CACHE/factor_cache_manifest.json"; exit 2; }

"$PY" -u scripts/exp/exp_nam_gate.py \
  --stocks 6000 --years 9 --end "$END" --disable-gate --target returns \
  --skip-baseline --cache-dir "$CACHE" \
  --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
  --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
  --y-scale 2 --expert-hidden 16 --lr 2e-3 \
  --store-dtype fp16 --seeds 42,11,23,37 \
  --save-model-dir "models/nam_gate/T122_prod9y" \
  --plot-dir "diagnose_output/nam_gate_T122" \
  --output "diagnose_output/T122_prod9y.json"
rc=$?
echo "[T122] exit=$rc  $(date '+%H:%M:%S')"
exit $rc
