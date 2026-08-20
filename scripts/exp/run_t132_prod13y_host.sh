#!/usr/bin/env bash
# T132（③）：**生产模型训至最新日 × 全 13 年窗** —— DataLoader 主机驻留把显存天花板拿掉后
#           第一次真用它。这是 T122 唯一被迫妥协的那一项。
#
# 为什么值得跑
#   T122 的窗口是 `--years 9`，注释里写得很清楚：**纯显存硬约束，不是选择** ——
#     13y(2013-08→2026-08) 12.11M 样本 → fp16 5.05 GiB + 非特征开销 1.87 GiB = 6.92 GiB
#     本机 RTX 4050 Laptop 只有 6140 MiB ⇒ 超配会被 WDDM 静默页到内存，实测慢 12.5x。
#   `--store-device host` 把特征放进 pinned 主机内存、侧流异步取，实测只慢 1.8%
#   （66.01 vs 64.83 ms/chunk），于是 13y 变成可训的。
#   ⇒ 本臂与 T122 的差别**只有窗口长度一项**（9y → 13y），是干净的单变量对比。
#
# 内存预算（必须先算，主机驻留换的是内存压力）
#   pinned fp16 特征  12.11M × 224 × 2B ≈ 5.4 GiB
#   取数阶段的 float32 面板（prepare_dataset 产物，训练开始前释放）≈ 10.9 GiB
#   本机 31.7 GiB 总内存 / 启动时可用约 14 GiB ⇒ 取数阶段是瓶颈，不是训练阶段。
#   若在 prepare_dataset 阶段 OOM，退到 --years 11 再试；**不要**回去关 host 驻留。
#
# ── 预注册判定（出结果前写死）───────────────────────────────────────────────
#   Q1「多 4 年历史值不值」：T132 vs T122 **不能配对**（换窗就换考卷，见
#      [[holdout-ic-not-comparable-across-windows]]）。唯一诚实的比法是把两批存档
#      放到**同一个窗口同一个 holdout** 上：
#        python scripts/exp/eval_models_on_window.py --train-fraction 0.8 --select-holdout 0.4
#      判据：T132 四种子在同窗 holdout 上的 IC 中位 ≥ T122 + 0.005，且**跌日 IC 不更差**。
#   Q2「选型段 regime 偏置有没有更糟」：13y 窗的后 20% ≈ 2024-01→2026-08，
#      仍是偏牛。必看 holdout **跌日 rank IC**；若显著低于 T122，说明多给的历史
#      没能抵消选型段偏置，据实记录，不许只看合并 IC 就晋级。
#   Q3「主机驻留的代价是否如实」：日志里 `_DayBatchLoader` 会打印驻留决策与原因，
#      对比 T122 的 ms/chunk。若慢超过 5%（而非 1.8%），说明 13y 窗上的取数瓶颈
#      与微基准不同，据实记录 —— 这正是 [[chunk-days-speedup-is-modest]] 那条教训。
#
# 与 T122 的差异清单（只有两项）
#   --years 9 → 13
#   新增 --store-device host
# 其余全同：--disable-gate --lambda-lb 0 --drop-groups forecast --select-holdout 0.4
#          --epochs 60 --min-epochs 20 --chunk-days 4 --y-scale 2 --expert-hidden 16
#          --lr 2e-3 --store-dtype fp16 --seeds 42,11,23,37
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-18-idxrel"
END="2026-08-10"

[ -f "$CACHE/factor_cache_manifest.json" ] || {
  echo "[ABORT] 缺缓存清单 $CACHE/factor_cache_manifest.json"; exit 2; }

"$PY" -u scripts/exp/exp_nam_gate.py \
  --stocks 6000 --years 13 --end "$END" --disable-gate --target returns \
  --skip-baseline --cache-dir "$CACHE" \
  --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
  --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
  --y-scale 2 --expert-hidden 16 --lr 2e-3 \
  --store-dtype fp16 --store-device host --seeds 42,11,23,37 \
  --save-model-dir "models/nam_gate/T132_prod13y" \
  --plot-dir "diagnose_output/nam_gate_T132" \
  --output "diagnose_output/T132_prod13y.json"
rc=$?
echo "[T132] exit=$rc  $(date '+%H:%M:%S')"
exit $rc
