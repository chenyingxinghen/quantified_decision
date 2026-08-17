#!/usr/bin/env bash
# T105：**生产载体终审第二轮** —— 全量池 blend（xgb+lgb 等权）vs 全量池 NAM。
#
# 上游结论链：
#   T095：800 只树 vs NAM → 门2 判不出，门3 平局判给 NAM（不是因为它选股强）。
#   T101/T103：池规模翻转模型排序；全量单模型树两窗皆负 + 胜率更低，已否证出局。
#   T103：全量 blend s42 熊 +26.72pp（超 3/4 个 NAM 种子）、牛 +0.11pp（超 2/4），
#         唯一两窗都不落后的树侧候选 —— 但 n=1 无证据，本轮扩种子。
#
# 本轮（2026-08-17 用户指示「2种子全量训练和回测验证」）：
#   已有 s42，再训 s11/s23 → blend n=3 对 NAM n=3（同种子号配对，s37 不训）。
#   ⚠ 种子号在两个模型族之间没有因果联系，「配对」只是沿用 T095 口径；
#     σ_paired ≈ √(σ_blend² + σ_nam²)，判读时按这个理解。
#
# **判据（预注册，沿用 T095 三门，出结果后照判不商量）**：
#   门1 无 regime 债：blend 在两个窗口都不输 NAM 超过窗内 MDE(n=3)。
#   门2 净增益：满足门1 且两窗合计 Δ > 合并 MDE。
#   门3 平局：β 更接近 1、回撤更浅者进生产。
#   n=3 的 MDE 会很大（T095 n=4 时约 80pp）——大概率落在门3。
#   若门1 就不过（blend 有 regime 债），维持 NAM，混合轴永久关闭。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
W1_S="2022-09-05"; W1_E="2024-08-05"   # 熊：基准 −13.82%
W2_S="2024-08-05"; W2_E="2026-08-05"   # 牛：基准 +95.64%
LOG=diagnose_output/T105_driver.log
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

for S in 11 23; do
  MARK="diagnose_output/.t105_trees_s${S}"
  if [ ! -f "$MARK" ]; then
    say "训练全量池 xgboost+lightgbm seed=$S（5480 只，截止 2022-09-05）"
    "$PY" -u scripts/train_model.py --end 2022-09-05 --stocks 6000 \
        --models xgboost lightgbm --cache-dir "$CACHE" --seed "$S" \
        --skip-cache-update --no-update-latest \
        > "diagnose_output/T105_trees_s${S}_train.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { say "训练失败 s$S rc=$rc"; tail -20 "diagnose_output/T105_trees_s${S}_train.log"; exit 1; }
    D=$(ls -dt models/xl_7d_17y_5480s_*_s${S} 2>/dev/null | head -1)
    [ -z "$D" ] && { say "找不到 s$S 全量树目录"; exit 1; }
    echo "$D" > "$MARK"; say "  → $D"
  fi
  D=$(cat "$MARK")
  for W in "$W1_S $W1_E" "$W2_S $W2_E"; do
    set -- $W
    mk="diagnose_output/.bt105_blend_s${S}_${1}.done"
    out="diagnose_output/T105_bt_blend_s${S}_${1}.log"
    [ -f "$mk" ] && { say "跳过 blend s$S $1（已完成）"; continue; }
    say "回测 blend s$S  $1 → $2"
    "$PY" -u scripts/run_backtest.py --start "$1" --end "$2" --cache-dir "$CACHE" \
        --tag "T105_blend_s${S}" \
        --model "$D/xgboost_factor_model.pkl" \
        --ensemble-model "$D/lightgbm_factor_model.pkl" > "$out" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then say "回测失败 s$S $1 rc=$rc"; tail -20 "$out"; else touch "$mk"; fi
  done
done
say "全部完成"
"$PY" -u scripts/archive/t_series/analyze_t105.py | tee -a "$LOG"
