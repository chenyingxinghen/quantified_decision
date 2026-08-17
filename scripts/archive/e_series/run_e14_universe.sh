#!/usr/bin/env bash
# T084 / E14：训练股票池的代表性 —— 零训练，只换评估股票池
#
# 发现：训练用 codes[:800]，全是 000/001/002 前缀（深市主板+中小板，≤002205）；
# 回测却在全市场 5480 只上选股，实际成交里只有 12~15% 落在训练池内 ——
# 85% 以上的持仓是模型训练时从没见过的板块（沪市 600/601/603/688 约 40%，创业板 300/301 约 20%）。
#
# 本轮只问一件事：**我们一直在优化的那把 IC 尺子，量的是不是错的面板？**
# 用同一批基线模型（T045 / T068_seed*），分别在训练口径股票池和「训练完全没见过的
# 800 只」上算同窗口 IC。若后者显著更低 ⇒ E7~E13 全部轴的判定都建立在一个只占
# 实际成交 13% 的面板上，方向必须改。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e14.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

MODELS="--model base_s42=models/nam_gate/T045
        --model base_s11=models/nam_gate/T068_seed11
        --model base_s23=models/nam_gate/T068_seed23
        --model base_s37=models/nam_gate/T068_seed37"

echo "########## 训练没见过的 800 只（holdout） ##########"
"$PY" scripts/exp/diag_ic_by_regime.py $MODELS \
  --stocks 800 --years 13 --end 2026-08-05 --topk 40 \
  --code-mode holdout_random \
  --out diagnose_output/T084_ic_holdout.json

echo "########## 全市场随机 800 只（代表性样本） ##########"
"$PY" scripts/exp/diag_ic_by_regime.py $MODELS \
  --stocks 800 --years 13 --end 2026-08-05 --topk 40 \
  --code-mode random \
  --out diagnose_output/T084_ic_random.json

echo "=== T084 E14 DONE ==="
