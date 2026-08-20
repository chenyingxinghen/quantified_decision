#!/usr/bin/env bash
# T120：回测验证 —— 用 OOS 两窗回测复核两件事（2026-08-19）
#
# ① **门控轴关闭的回测复核**（今天的主结论）。T118/T119 在无偏 holdout 上
#    双双 0/4、Δ中位 −0.033/−0.028、跌日 0/4 触发否决门。台账有先例：T108/T111
#    做过「IC 与回测最终一致」的交叉验证，也有过反例 —— T107 单种子 s42 出现过
#    「IC 跌但回测两窗都不输」。所以终审关轴前值得再看一眼回测方向是否一致。
# ② **生产载体 T115_idxrel_s42 的回测基线**。它是**按用户判断晋级**的（3/4，
#    未过 4/4 预注册门），且已接入自动化，但**从未回测过**。T113 是它的 219 列前身。
#
# ⚠ 分辨率警告：n=4 配对回测 MDE 实测约 80pp（[[backtest-mde-is-80pp]]），
#   本批**每臂只有 1 个种子（s42）**，分辨率比那还差。因此这批数据
#   **只能用于否决/发现灾难性劣化与形状异常（β、胜率、跌日行为），
#   绝不能用来晋级或反推「谁多赚了几 pp」**。判定权仍在验证 Rank IC。
#
# 口径（与 T099/T103 那批一致，全部取 strategy_config 默认，不传覆盖）：
#   MAX_POSITIONS=5  MIN_CONFIDENCE=0  RISK_MIN_PRICE=1.0  RISK_EXCLUDE_ST=True
#   SELECTOR_MARKETS=['sh_main','sz_main']
# 缓存**不显式指定**：每个存档按自身 factor_cache_manifest.json 解析 ——
#   T113 → factors_cache_2026-08-14-fwdadjust（219 列）
#   T115/T118/T119 → factors_cache_2026-08-18-idxrel（224 列）
#   两个缓存的共有 219 列同公式同复权口径，可配对。
#
# ⚠ 回测 stdout 必须 UTF-8，否则会在 5 分钟预加载之后才崩（[[backtest-stdout-gbk-trap]]）。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
W1_S="2022-09-05"; W1_E="2024-08-05"   # 熊：基准 −13.82%（主判窗）
W2_S="2024-08-05"; W2_E="2026-08-05"   # 牛：基准 +95.64%
LOG=diagnose_output/T120_driver.log

say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# 顺序刻意如此：生产载体先跑完两窗，最重要的数字最早落地。
ARMS="T115_idxrel_s42 T119_groupnorm_s42 T118_datagroups_s42 T113_fp16_s42"

for A in $ARMS; do
  M="models/nam_gate/${A}/nam_gate_factor_model.pkl"
  if [ ! -f "$M" ]; then say "缺存档：$M —— 跳过"; continue; fi
  for W in "$W1_S $W1_E" "$W2_S $W2_E"; do
    set -- $W
    MARK="diagnose_output/.bt120_${A}_${1}.done"
    OUT="diagnose_output/T120_bt_${A}_${1}.log"
    [ -f "$MARK" ] && { say "跳过 $A $1（已完成）"; continue; }
    say "回测 $A  $1 → $2"
    "$PY" -u scripts/run_backtest.py --start "$1" --end "$2" \
        --tag "T120_${A}" --model "$M" > "$OUT" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then say "失败 $A $1 rc=$rc"; tail -25 "$OUT"; else touch "$MARK"; fi
  done
done
say "全部完成"
