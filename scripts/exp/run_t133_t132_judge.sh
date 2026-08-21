#!/usr/bin/env bash
# T133/T132 判定串行驱动。两者都要建自己的面板（各约 10 GB 主机内存），
# 必须排队而不是并发 —— 并发会把 31.7 GB 的机器逼进交换。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"

M133=""
for YS in 1 2 4 8; do
  for SD in 42 11; do
    M133="${M133}${M133:+,}models/nam_gate/T133_yscale${YS}_s${SD}"
  done
done

echo "########## 判定开始 $(date '+%F %H:%M:%S') ##########"

echo "===== ① T133 剂量扫描裁决（13y→2022-09-05，与 T115 同卷）====="
"$PY" -u scripts/exp/judge_yscale_dose.py \
  --models "$M133" --baseline-arm 2 \
  --years 13 --end 2022-09-05 \
  --topk 20 --extra-topk 1,5,50 --cross-pool-n 200 \
  --cache-npz diagnose_output/T133_groupsums.npz \
  --out diagnose_output/T133_yscale_dose.json
echo "[judge T133] exit=$? $(date '+%F %H:%M:%S')"

echo
echo "===== ② T132 vs T122 同窗同 holdout（9y→2026-08-10）====="
# 换窗就换考卷：T132 是 13y 窗、T122 是 9y 窗，各自的 holdout 落在不同日历段，
# 直接比 rank_ic_holdout 无效。这里把两批 8 个存档放同一张卷子上重算。
#
# ⚠⚠ 事后发现的混淆（原注释写反了，特此更正）：
#   `--folds 0.8:1.0` 是**比例**切分，窗口越长切点越早：
#     T132（13y）实际训到 **2024-08-27**
#     T122（ 9y）实际训到 **2025-01-20**
#   两者数据结束日相同（2026-08-10），但 **T122 多了约 5 个月的近期训练数据**。
#   所以这张卷子偏向 **T122**，不是 T132。原注释"该窗对 T132 更有利"是错的。
#   ⇒ 本次比较把"窗口长度"和"训练时效"混在一起了。要拆开必须让两者**切点相同**：
#      13y 窗需 --folds 0.881:1.0 才落在 2025-01-20。
#   选型污染另说，这条是干净的：两批的 checkpoint 选型段都不含本卷 holdout
#   （T132 自己 holdout 183 日 ⊃ 本卷 holdout 145 日，同锚于 2026-08-10）。
M132=""
for B in T132_prod13y T122_prod9y; do
  for SD in 42 11 23 37; do
    M132="${M132}${M132:+,}models/nam_gate/${B}_s${SD}"
  done
done
"$PY" -u scripts/exp/eval_models_on_window.py \
  --models "$M132" --years 9 --end 2026-08-10 \
  --train-fraction 0.8 --select-holdout 0.4 \
  --out diagnose_output/T132_vs_T122_same_window.json
echo "[eval T132] exit=$? $(date '+%F %H:%M:%S')"

echo "########## 判定结束 $(date '+%F %H:%M:%S') ##########"
