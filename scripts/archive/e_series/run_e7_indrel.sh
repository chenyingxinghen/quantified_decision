#!/usr/bin/env bash
# T076 / E7：行业内相对分位（``<base>__ind``）—— 回到「让模型排得更准」的 alpha 轴
#
# 为什么是这条：现有 219 列**全部**是全市场截面量，训练前又统一做当日全市场 rank
# 归一化，所以模型无法区分「这只票强」和「它所在的行业强」。A 股动量/估值的截面差异
# 有很大一部分是行业共同驱动，剥掉它才剩个股 alpha。这是一条新信息轴，且
# 不需要新数据、不需要重建 13.85GB 因子缓存（在内存里按 (date, industry) 求分位）。
#
# 判定口径（本轮改掉评估器，这是效率关键）：
#   打分层的改动**用验证 Rank IC 判定，不跑回测**。理由见台账「今日效应量分层」：
#   回测+随机零假设的 MDE 在 K=40 上还有 1.64 z，而 Rank IC 是直接、同标签可比、
#   零回测成本的排序精度度量。T073 已经示范过：门控回测赢了但 IC 0/4 下降——
#   那正是「规则让回测好看」而非 alpha。只有 IC 4/4 上升才有资格进回测。
#   基线：T045/T068_seed* 的 best_val_rank_ic_on_label
#   （s42 0.09685 / s11 0.08868 / s23 0.09384 / s37 0.09225，同标签、同折、同 target）。
#
# 变量隔离：与基线**唯一**差别是 --industry-relative core（+18 列，新增 indrel 族）。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e7_indrel.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

for SEED in 42 11 23 37; do
  OUT="models/nam_gate/T076_indrel_s${SEED}"
  JSON="diagnose_output/T076_indrel_s${SEED}.json"
  LOG="diagnose_output/T076_indrel_s${SEED}_train.log"
  if [ -f "$JSON" ]; then echo "[SKIP] $JSON 已存在"; continue; fi
  echo "[TRAIN] seed=$SEED -> $OUT"
  "$PY" scripts/exp/exp_nam_gate.py \
    --stocks 800 --years 13 --end 2022-09-05 \
    --disable-gate \
    --target returns \
    --y-scale 2 --lambda-lb 0 \
    --epochs 60 --min-epochs 20 --skip-baseline \
    --folds "0.8:1.0" --seed "$SEED" \
    --allow-degenerate-downside-risk \
    --industry-relative core \
    --save-model-dir "$OUT" \
    --output "$JSON" \
    > "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL TRAIN] seed=$SEED rc=$rc"; tail -20 "$LOG"; }
  # T043 铁律：即使本轮只看 IC，模型目录也必须带 norm_stats.pkl，否则晋级时无法直接回测
  [ -f "$OUT/norm_stats.pkl" ] || echo "[WARN] $OUT 缺 norm_stats.pkl"
done
echo "=== T076 E7 INDREL TRAIN DONE ==="
"$PY" scripts/archive/e_series/analyze_e7.py || true
