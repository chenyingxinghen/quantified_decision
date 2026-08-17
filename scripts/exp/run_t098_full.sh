#!/usr/bin/env bash
# T098：把 T095/T096 定下的 NAM 冠军配置放大到**全量股票池**，并留下可回测的存档。
#
# 配置来源（不再调参，三个轴都关了）：
#   --y-scale 2 --expert-hidden 16 --lr 2e-3   ← T096 四臂无人晋级，base 保持冠军
#   --disable-gate --lambda-lb 0               ← T095 终审用的就是纯加性 NAM
#   --drop-groups forecast                     ← 预告族单因子 IC 真但混进打分无增益
#
# **为什么先跑 2022-09-05 截止而不是直接训到最新**：
#   1. 全量池上超参优劣可能与 800 只不同 —— T096 的结论建立在「排序不随池规模翻转」
#      这个**假设**上。截止日对齐才能和 800 只的 holdout 0.08229 直接比。
#   2. 这一批存档正好可以跑 T095 的两个 OOS 窗口（熊 2022-09→2024-08、
#      牛 2024-08→2026-08），做回测反馈验证。训到最新的模型没有 OOS 可言。
#   生产存档（训到 2026-08-10）是第二阶段，等这批的 IC 和回测都站得住再说。
#
# 判据：全量 holdout 折均 IC 不显著低于 800 只的 0.08229；跌日 IC 不塌。
#       若排序翻转（全量更差），如实报告并停下来讨论，不要硬推进生产。
#
# **为什么只跑 0.8:1.0 单折**（2026-08-15 18:50 用户决定，原计划 3 折）：
#   实测约 6 分钟/epoch（GPU 100%、显存 5713/6141MiB 打满，已是算力瓶颈，
#   chunk-days 帮不上），按早停约 35 epoch 算单折 3.5h、单种子 3 折 10.5h、
#   4 种子 42h。缩到单折后约 14h。
#   取舍：**保留 4 种子配对功率（判定的关键），牺牲折平均降噪**。单折 σ 约放大
#   √3 倍，所以只能判「全量是否显著低于 800 只的 0.08229」这种粗问题，
#   不追 0.001 量级。选 0.8:1.0 是因为它离 2022-09-05 最近，与 OOS 回测衔接最紧。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"

# --stocks 6000 > 池内 5446 只，等于「全部」；写死数字而不是 0 是因为
# 上游是 codes[:n] 切片，没有「全取」的哨兵值。
COMMON="--stocks 6000 --years 13 --end 2022-09-05 --disable-gate --target returns \
--skip-baseline --allow-degenerate-downside-risk --cache-dir $CACHE \
--folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
--epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
--y-scale 2 --expert-hidden 16 --lr 2e-3"

for S in 42 11 23 37; do
  J="diagnose_output/T098_full_s${S}.json"
  [ -f "$J" ] && { echo "[SKIP] $J"; continue; }
  echo "[TRAIN] full seed=$S  $(date '+%H:%M:%S')"
  "$PY" -u scripts/exp/exp_nam_gate.py $COMMON --seed "$S" \
    --save-model-dir "models/nam_gate/T098_full_s${S}" \
    --plot-dir "diagnose_output/nam_gate_T098_s${S}" \
    --output "$J" > "diagnose_output/T098_full_s${S}_train.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] s$S rc=$rc"; tail -15 "diagnose_output/T098_full_s${S}_train.log"
  else
    echo "[OK] s$S  $(date '+%H:%M:%S')"
  fi
done

echo "[DONE] $(date '+%H:%M:%S')"
# 与 800 只的 T096 base 横比，但**必须同折**（base 是 3 折，这里只有 0.8:1.0）
"$PY" -u - <<'PYEOF'
import json, os, numpy as np
FOLD, SEEDS = '80-100%', (42, 11, 23, 37)
def ho(prefix, arm, s):
    for d in ('diagnose_output', 'diagnose_output/iter_output'):
        p = os.path.join(d, f'{prefix}_{arm}_s{s}.json')
        if os.path.isfile(p):
            f = json.load(open(p, encoding='utf-8'))['folds'].get(FOLD, {}).get('nam_gate')
            return None if not f else f
    return None
print(f"{'seed':>6} {'800只 holdout':>14} {'全量 holdout':>14} {'Δ':>10}"
      f" {'800跌日':>9} {'全量跌日':>9}")
d = []
for s in SEEDS:
    a, b = ho('T096', 'base', s), ho('T098', 'full', s)
    if not (a and b):
        print(f'{s:>6}   （缺）'); continue
    v = b['rank_ic_holdout'] - a['rank_ic_holdout']; d.append(v)
    print(f'{s:>6} {a["rank_ic_holdout"]:>14.5f} {b["rank_ic_holdout"]:>14.5f} {v:>+10.5f}'
          f' {a["regime"]["down_days"]["rank_ic"]:>9.5f}'
          f' {b["regime"]["down_days"]["rank_ic"]:>9.5f}')
if len(d) > 1:
    d = np.asarray(d); mde = 2.0 * d.std(ddof=1) / np.sqrt(len(d))
    print(f'\n  {int((d > 0).sum())}/{len(d)} 上升，Δ 中位 {np.median(d):+.5f}，'
          f'配对 MDE ≈ {mde:.5f}')
    print('  判读：|Δ| < MDE ⇒ 池规模没有翻转结论，可以推进生产；'
          'Δ 显著为负 ⇒ 800 只上的超参选择不适用于全量，停下来讨论。')
PYEOF
