#!/usr/bin/env bash
# T103：**全量池树模型的回测检验** —— 单种子，两个 regime 窗口。
#
# **为什么必须回测而不是看 IC**（2026-08-17 用户第二次指出，采纳）：
#   lambdarank 优化的是 ndcg 头部截断，整截面 Rank IC 度量的是全体排序质量。
#   一个「只把前 20 名排对、中后段乱排」的模型 IC 会平平，但那正是生产要的东西。
#   T101 用 IC 判全量 lgb（0.09081）不如全量 NAM（0.11114），这个读数**不能**
#   直接推出「树的头部选股更差」—— 那是两件事。回测的持仓就是头部 20 只，
#   是唯一直接检验头部能力的口径。
#   ⚠ 已知代价：n=1 种子的回测没有配对功率（[[backtest-mde-is-80pp]] 实测
#   n=4 时 MDE 约 80pp）。所以这一轮**只看量级和方向**，不做晋级判定；
#   若树明显赢（如某窗口超额高出 30pp 以上），再扩种子做正式判定。
#
# **为什么以前没测过**：T095 的 blend 臂用的是 `train_model.py --stocks 800` 训的树，
#   而 T098/T099 的 NAM 是全量 5450 只 —— 两者从来没在同一池规模上回测比过。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
W1_S="2022-09-05"; W1_E="2024-08-05"   # 熊：基准 −13.82%
W2_S="2024-08-05"; W2_E="2026-08-05"   # 牛：基准 +95.64%
LOG=diagnose_output/T103_driver.log
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

MARK="diagnose_output/.t103_trees_full"
if [ ! -f "$MARK" ]; then
  say "训练全量池 xgboost+lightgbm（5480 只，截止 2022-09-05，seed 42）"
  "$PY" -u scripts/train_model.py --end 2022-09-05 --stocks 6000 \
      --models xgboost lightgbm --cache-dir "$CACHE" --seed 42 \
      --skip-cache-update --no-update-latest \
      > diagnose_output/T103_trees_train.log 2>&1
  rc=$?
  [ $rc -ne 0 ] && { say "树训练失败 rc=$rc"; tail -25 diagnose_output/T103_trees_train.log; exit 1; }
  D=$(ls -dt models/xl_*_s42 2>/dev/null | head -1)
  [ -z "$D" ] && { say "找不到全量树目录"; exit 1; }
  echo "$D" > "$MARK"; say "  → $D"
fi
D=$(cat "$MARK")

# 三个臂：lgb 单模型（头部能力的直接检验）、xgb 单模型、等权混合
run_bt () {  # run_bt <臂名> <窗口起> <窗口止> <模型参数...>
  local arm="$1" ws="$2" we="$3"; shift 3
  local mk="diagnose_output/.bt103_${arm}_${ws}.done"
  local out="diagnose_output/T103_bt_${arm}_s42_${ws}.log"
  [ -f "$mk" ] && { say "跳过 $arm $ws（已完成）"; return; }
  say "回测 $arm  $ws → $we"
  "$PY" -u scripts/run_backtest.py --start "$ws" --end "$we" --cache-dir "$CACHE" \
      --tag "T103_${arm}_s42" "$@" > "$out" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "失败 $arm $ws rc=$rc"; tail -20 "$out"; else touch "$mk"; fi
}

for W in "$W1_S $W1_E" "$W2_S $W2_E"; do
  set -- $W
  run_bt lgb   "$1" "$2" --model "$D/lightgbm_factor_model.pkl"
  run_bt xgb   "$1" "$2" --model "$D/xgboost_factor_model.pkl"
  run_bt blend "$1" "$2" --model "$D/xgboost_factor_model.pkl" \
                         --ensemble-model "$D/lightgbm_factor_model.pkl"
done
say "全部完成"
