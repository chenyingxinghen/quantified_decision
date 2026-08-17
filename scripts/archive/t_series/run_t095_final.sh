#!/usr/bin/env bash
# T095：**生产载体终审** —— xgb+lgb 等权混合 vs NAM，4 种子配对 × 2 个 regime 窗口。
#
# 为什么是这两个臂：T092/T094 证明单模型各自押死了一种风格（gain 重要性 Spearman
# ρ=0.457，market_cap 占比 17.2% vs 1.1%），而「何时该用谁」不可预测（ΔIC_t 折外
# R²=−0.19，月均离散度已低于纯噪声理论值）。条件混合关轴后剩下两个候选：
# 静态等权混合（吃分散化）与 NAM（结构上学不了交互，风格暴露最浅）。
#
# 判据（2026-08-15 用户预注册，出结果后按此判，不再商量）：
#   1. 无 regime 债：两个窗口都不输对手超过窗内 MDE（n=4 配对 ≈13pp）。
#      一窗大赢一窗大输 = 风格彩票，直接出局，不管合计多好看。
#   2. 有净增益：满足 1 的前提下，两窗合计超额为正且过噪声门。
#   3. 平局判给 β 更接近 1、回撤更浅的那个。
#
# ⚠ 必须跑在 preclose 复权口径上（handler 已于 2026-08-15 00:24 切换）。
#   T092 那批回测跑在旧的 adjust_factor 表口径上（覆盖率 42.5%），已作废，勿混用。
# ⚠ 回测 stdout 必须 UTF-8，否则会在 5 分钟预加载之后才崩（见 [[backtest-stdout-gbk-trap]]）。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
TRAIN_END="2022-09-05"
W1_S="2022-09-05"; W1_E="2024-08-05"   # 熊：基准 −13.82%
W2_S="2024-08-05"; W2_E="2026-08-05"   # 牛：基准 +95.64%
SEEDS="42 11 23 37"
LOG=diagnose_output/T095_driver.log

say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ── 1. 生产配置的树，每个种子一对（xgb+lgb 落在同一个 xl_* 目录）────────
for S in $SEEDS; do
  MARK="diagnose_output/.t095_trees_s${S}"
  [ -f "$MARK" ] && { say "跳过树训练 s$S（已完成）"; continue; }
  say "训练 xgboost+lightgbm 生产配置 seed=$S（截止 $TRAIN_END）"
  "$PY" -u scripts/train_model.py --end "$TRAIN_END" --stocks 800 \
      --models xgboost lightgbm --cache-dir "$CACHE" --seed "$S" \
      --skip-cache-update --no-update-latest \
      > "diagnose_output/T095_trees_s${S}_train.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { say "树训练失败 s$S rc=$rc"; tail -25 "diagnose_output/T095_trees_s${S}_train.log"; exit 1; }
  # --seed 会给目录加 _sN 后缀，据此定位而不是靠 mtime（同分钟落盘会撞车）
  D=$(ls -dt models/xl_*_s${S} 2>/dev/null | head -1)
  [ -z "$D" ] && { say "找不到 s$S 的模型目录"; exit 1; }
  echo "$D" > "$MARK"
  say "  → $D"
done

# ── 2. 回测：混合 / NAM，各 4 种子 × 2 窗口 ─────────────────────────────
run_bt () {  # run_bt <臂> <种子> <窗口起> <窗口止> <额外参数...>
  local arm="$1" s="$2" ws="$3" we="$4"; shift 4
  local done_mark="diagnose_output/.bt095_${arm}_s${s}_${ws}.done"
  local out="diagnose_output/T095_bt_${arm}_s${s}_${ws}.log"
  [ -f "$done_mark" ] && { say "跳过 $arm s$s $ws（已完成）"; return; }
  say "回测 $arm  seed=$s  $ws → $we"
  "$PY" -u scripts/run_backtest.py --start "$ws" --end "$we" --cache-dir "$CACHE" \
      --tag "T095_${arm}_s${s}" "$@" > "$out" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "回测失败 $arm s$s $ws rc=$rc"; tail -20 "$out";
  else touch "$done_mark"; fi
}

for S in $SEEDS; do
  D=$(cat "diagnose_output/.t095_trees_s${S}")
  NAM="models/nam_gate/T092_nam_s${S}"
  # NAM 存档链是并行跑的，可能还没轮到这个种子。等而不是跳过 —— 跳过会让
  # 该种子的 nam 臂永久缺失，配对判定退化成 n=3，静默削弱功率。
  waited=0
  while [ ! -f "$NAM/nam_gate_factor_model.pkl" ] && [ $waited -lt 5400 ]; do
    [ $waited -eq 0 ] && say "等待 NAM s$S 存档就绪…"
    sleep 60; waited=$((waited + 60))
  done
  if [ ! -f "$NAM/nam_gate_factor_model.pkl" ]; then
    say "NAM s$S 等待超时（90 分钟），该种子的 nam 臂缺失 —— 判决会退化成 n=3"
    NAM=""
  fi
  for W in "$W1_S $W1_E" "$W2_S $W2_E"; do
    set -- $W
    run_bt blend "$S" "$1" "$2" --model "$D/xgboost_factor_model.pkl" \
                                --ensemble-model "$D/lightgbm_factor_model.pkl"
    [ -n "$NAM" ] && run_bt nam "$S" "$1" "$2" --model "$NAM"
  done
done

say "全部完成"
"$PY" -u scripts/archive/t_series/analyze_t095.py
