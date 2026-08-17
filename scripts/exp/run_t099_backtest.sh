#!/usr/bin/env bash
# T099：全量池 NAM 存档的 OOS 回测反馈验证（4 种子 × 2 个 regime 窗口）。
#
# 上游：T098 全量池（5450 只）单折 0.8:1.0 × 4 种子，同折无偏 holdout 对 800 只
#       **4/4 上升、Δ 中位 +0.0426 ≈ 7× 配对 MDE**，涨日跌日各 4/4 正。
#
# **为什么 IC 过了还要回测**：全量池的 IC 抬升有一部分可能来自「小盘/低流动性股票
# 更好排」—— 截面变宽本身就让 Spearman 更稳。回测是唯一能检验这部分是否**可交易**
# 的手段（涨跌停、停牌、冲击成本、ST 过滤都在回测里）。
#
# **判据（沿用 T095 预注册的三门，但这次是「验收」不是「选型」）**：
#   门1 无 regime 债：两个窗口都不显著落后于 T095 的 NAM 800 只臂。
#   门2 摩擦未吃掉增益：全量臂两窗合计超额 ≥ 800 只臂（同种子配对）。
#   门3 平局看 β 与回撤。
#   ⚠ n=4 配对回测的 MDE 实测约 80pp（见 [[backtest-mde-is-80pp]]）——
#     所以这里**只能否决不能晋级**：查有没有灾难性劣化，不追几 pp 的差异。
#
# ⚠ 回测 stdout 必须 UTF-8，否则会在 5 分钟预加载之后才崩（[[backtest-stdout-gbk-trap]]）。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
W1_S="2022-09-05"; W1_E="2024-08-05"   # 熊：基准 −13.82%
W2_S="2024-08-05"; W2_E="2026-08-05"   # 牛：基准 +95.64%
LOG=diagnose_output/T099_driver.log

say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

for S in 42 11 23 37; do
  M="models/nam_gate/T098_full_s${S}/nam_gate_factor_model.pkl"
  if [ ! -f "$M" ]; then say "缺存档 s$S：$M —— 跳过"; continue; fi
  for W in "$W1_S $W1_E" "$W2_S $W2_E"; do
    set -- $W
    MARK="diagnose_output/.bt099_s${S}_${1}.done"
    OUT="diagnose_output/T099_bt_full_s${S}_${1}.log"
    [ -f "$MARK" ] && { say "跳过 s$S $1（已完成）"; continue; }
    say "回测 full-NAM seed=$S  $1 → $2"
    "$PY" -u scripts/run_backtest.py --start "$1" --end "$2" --cache-dir "$CACHE" \
        --tag "T099_full_s${S}" --model "$M" > "$OUT" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then say "失败 s$S $1 rc=$rc"; tail -20 "$OUT"; else touch "$MARK"; fi
  done
done
say "全部完成"
