#!/usr/bin/env bash
# T134：**y_scale 剂量轴的回测机制验证**（不是晋级实验）
#
# ⚠⚠ 这批回测**没有晋级资格**，不可能让 y4/y8 翻盘。
#   IC 优先协议里回测只能否决不能晋级（[[alpha-not-rules-ic-first-protocol]]），
#   而 y4/y8 已在 T133 被**跌日否决门**拒了（0/2 为正），y1 被证伪读数拒了。
#   本批唯一的目的是：**检验机制说法对不对**。
#
# 待检验的机制假设 H
#   T133 观察到 y_scale 拧大时「涨日头部略好、跌日头部塌」，我据此说
#   「y_scale 是一个**静态**风险偏好旋钮」。若此说为真，可证伪预测是：
#     H1) **β 随 y_scale 单调上升**（这是核心预测，也是最容易测准的一个）
#     H2) 牛市窗总收益随 y_scale 上升
#     H3) 熊市窗总收益随 y_scale 下降
#     H4) 上行捕获率↑、下行捕获率↑（都变大 = 更高杠杆式暴露，而非选股变好）
#   若 β 在两个种子上都**平坦或非单调**，则「风险偏好旋钮」这个说法被证伪 ——
#   那 T133 的涨跌日不对称就得换解释（最可能是有效样本量塌缩导致的纯噪声）。
#
# ⚠ 判据必须挂在 β 上，不能挂在收益上
#   n=2 种子的终值回测 MDE 约 80pp（[[backtest-mde-is-80pp]]），H2/H3 基本判不动，
#   只能看方向是否一致。而 β 是 ~480 个交易日的日频回归，标准误小两个数量级，
#   H1 才是真正可判的。**先看 β，收益只作辅证。**
#
# 样本外性质（这批的最大优势）
#   T133 全部训到 2020-12-18、验证段止于 2022-09-05。下面两个窗**起点就是 2022-09-05**，
#   所以是**既没训也没选型**的干净样本外，近四年。
#
# 口径：与 T120/T099/T103 完全一致 —— 全部取 strategy_config 默认，不传任何覆盖
#   MAX_POSITIONS=5  MIN_CONFIDENCE=0  RISK_MIN_PRICE=1.0  RISK_EXCLUDE_ST=True
#   产出目录名应含 minp1_nost_mp5；缺段即口径错配，不可与 T120 基线比。
# 缓存不显式指定：每个存档按自身 factor_cache_manifest.json 解析
#   （T133 全部绑定 factors_cache_2026-08-18-idxrel，已核对一致）。
#
# 免费的对照：
#   · y2_s42/s11 是 T115_s42/s11 的同代码重训（T133 已验证 val_ic 对到小数第 4 位），
#     而 T115_s42 在 T120 里用**同口径同两窗**回测过 ⇒ y2_s42 应与它接近，
#     这是对整条回测链路的一致性检查，不额外花钱。
#
# ⚠ stdout 必须 UTF-8，否则会在 5 分钟预加载之后才崩（[[backtest-stdout-gbk-trap]]）。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"

W1_S="2022-09-05"; W1_E="2024-08-05"   # 熊：基准 −13.82%
W2_S="2024-08-05"; W2_E="2026-08-05"   # 牛：基准 +95.64%
LOG=diagnose_output/T134_driver.log

say () { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

# 顺序：先把两个极端剂量（y8/y1）跑完两窗 —— β 的单调性检验只要两端就能先看出方向，
# 万一中途要中断，最有信息量的两个点已经落地。
ARMS=""
for YS in 8 1 4 2; do
  for SD in 42 11; do
    ARMS="${ARMS}${ARMS:+ }T133_yscale${YS}_s${SD}"
  done
done
say "队列 16 次回测（8 存档 × 2 窗），预计约 2h50m"
say "顺序: $ARMS"

for A in $ARMS; do
  M="models/nam_gate/${A}/nam_gate_factor_model.pkl"
  if [ ! -f "$M" ]; then say "缺存档：$M —— 跳过"; continue; fi
  for W in "$W1_S $W1_E" "$W2_S $W2_E"; do
    set -- $W
    MARK="diagnose_output/.bt134_${A}_${1}.done"
    OUT="diagnose_output/T134_bt_${A}_${1}.log"
    [ -f "$MARK" ] && { say "跳过 $A $1（已完成）"; continue; }
    say "回测 $A  $1 → $2"
    "$PY" -u scripts/run_backtest.py --start "$1" --end "$2" \
        --tag "T134_${A}" --model "$M" > "$OUT" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then say "失败 $A $1 rc=$rc"; tail -25 "$OUT"; else touch "$MARK"; fi
  done
done
say "全部完成 —— 判定用 scripts/exp/judge_yscale_backtest.py"
