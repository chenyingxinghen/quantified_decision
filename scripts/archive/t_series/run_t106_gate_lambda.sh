#!/usr/bin/env bash
# T106：**全量池 + 门控 + lambda_lb 防塌陷** —— 单种子一次。
#
# 用户 2026-08-17 指示：「开启门和 lambda，全量训一次」。
#
# **为什么这次值得试，尽管 T102/T104 在 800 只上判了不晋级**：
#   1. T102 的门控**塌陷了**（熵 0.118/2.398、status 族占比 98.4%），而那批跑的是
#      `--lambda-lb 0` —— 没有任何东西阻止退化。代码里 809 行本来就打印了警告
#      「最大族占比 > 0.60，建议上调 --lambda-lb」，我们当时没听。
#      所以 T102 测的是「塌陷的门控」，不是「门控」。本轮加 lambda_lb 才是真正的检验。
#   2. [[pool-size-flips-model-ranking]]：模型族的优劣会随池规模翻转（800 只上树赢
#      0.022、全量上 NAM 赢 0.025）。门控是**结构级**改动，同样必须在全量池上验。
#      800 只上不晋级不构成全量上不晋级的证据。
#
# ⚠ **技术约束**：lambda_lb > 0 与 --chunk-days > 1 互斥（门控正则依赖逐日 EMA 更新
#   节奏，分块会偷偷改 momentum，脚本会硬失败）。所以本轮回退到逐日路径，
#   单折约 5.5 小时（chunk-days 只省 1.1x，损失不大）。
#
# 臂：lb001 = lambda_lb 0.01（train_nam_gate 的函数默认值，即作者原意的量级）
#     lb01  = lambda_lb 0.1（十倍，若 0.01 仍塌陷则需要更强的约束）
# 先跑 lb001；若它的 max_share 仍 > 0.60 说明 0.01 压不住，再跑 lb01。
#
# **判据**：对照 T098_full_s42（同池同折同尺子，holdout 0.11114）。
#   ① 门控健康：max_share < 0.60 且权重的跨日变异系数 CV > 0.05
#      （T102 的 status 族 CV 只有 0.0025 = 静态重加权，不是 regime 路由）。
#      门控不健康则该臂无论 IC 多少都不算「门控有效」。
#   ② IC：Δ holdout ≥ +0.005 才值得扩种子；|Δ| < 0.002 关轴。
#   单种子无配对功率，只筛方向与量级。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
# 注意：不带 --chunk-days（与 lambda_lb 互斥），不带 --disable-gate（本轮要开门）
COMMON="--stocks 6000 --years 13 --end 2022-09-05 --target returns --skip-baseline \
--allow-degenerate-downside-risk --cache-dir $CACHE \
--folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
--epochs 60 --min-epochs 20 --y-scale 2 --expert-hidden 16 --lr 2e-3 \
--gate-mode softmax --seed 42"

run () {  # run <标签> <lambda_lb>
  local tag="$1" lb="$2"
  local J="diagnose_output/T106_${tag}_s42.json"
  [ -f "$J" ] && { echo "[SKIP] $J"; return; }
  echo "[TRAIN] $tag lambda_lb=$lb  $(date '+%H:%M:%S')"
  "$PY" -u scripts/exp/exp_nam_gate.py $COMMON --lambda-lb "$lb" \
    --save-model-dir "models/nam_gate/T106_${tag}_s42" \
    --plot-dir "diagnose_output/nam_gate_T106_${tag}" \
    --output "$J" > "diagnose_output/T106_${tag}_s42.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL] $tag rc=$rc"; tail -15 "diagnose_output/T106_${tag}_s42.log"; return; }
  echo "[OK] $tag  $(date '+%H:%M:%S')"
  # 塌陷自检：若 0.01 压不住，自动接力跑 0.1（省一次人工往返）
  local share
  share=$("$PY" -c "
import json,sys
d=json.load(open('$J',encoding='utf-8'))['folds']
r=[v['nam_gate'] for v in d.values() if isinstance(v,dict) and 'nam_gate' in v]
print(f\"{max(x.get('gate_max_share',1.0) for x in r):.4f}\")" 2>/dev/null || echo 1.0)
  echo "  → 门控最大族占比 $share（健康阈值 < 0.60）"
}

run lb001 0.01
S=$("$PY" -c "
import json
try:
    d=json.load(open('diagnose_output/T106_lb001_s42.json',encoding='utf-8'))['folds']
    r=[v['nam_gate'] for v in d.values() if isinstance(v,dict) and 'nam_gate' in v]
    print('collapsed' if max(x.get('gate_max_share',1.0) for x in r) > 0.60 else 'healthy')
except Exception: print('missing')")
echo "[CHECK] lb001 门控状态: $S"
[ "$S" = "collapsed" ] && { echo "[NEXT] 0.01 压不住塌陷，接力 lambda_lb=0.1"; run lb01 0.1; }

echo "[DONE] $(date '+%H:%M:%S')"
"$PY" -u scripts/archive/t_series/analyze_t106.py
