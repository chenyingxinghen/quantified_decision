#!/usr/bin/env bash
# T119：门控轴结构性修复②——**族输出截面标准化**，消掉 w·f 的平坦方向。
#
# 立论（T116 诊断）：
#   乘积 w_k·S_k 在 (w_k→c·w_k, S_k→S_k/c) 下完全不变 ⇒ 损失曲面存在零曲率方向 ⇒
#   门控权重在上面随噪声漂移，专家幅度默默把门控做的事抵消掉。
#   实测指纹：status 族 gate_std 4.45 > gate_mean 3.97（变异系数>1 = 未被数据约束），
#   而它恰好贡献最小（每列 0.0006，贡献越小方向越平）；再经 Σw_k=K 的零和约束，
#   status 独占 3.97 使其余 11 族被压到 0.62~0.78（普遍饿约 27%）。
#
# 机制已离线验证：把某族专家输出整体×3，
#   关闭 group_norm → 打分排序相关 0.8645（幅度改变排序 ⇒ w 可被 f 吸收）
#   开启 group_norm → 打分排序相关 1.0000（幅度锁死 ⇒ w 不可被吸收）
#
# 变量隔离：与 T116 的**唯一**差别是 --group-norm。面板同为 224 列手工分族、
#   门控同为 scalar + macro_m1m2_gap、超参与 dtype 全同。
#   （分族那条变量由 T118 单独测，不在本臂混入。）
#
# 预注册判定：
#   Q1「平坦方向是不是 T116 失败的原因」：T119 vs **T116**。
#      主看两件事：① 跌日 ΔIC ≥3/4 为正；② 门控是否稳定下来 ——
#      各族 gate_std/gate_mean 应普遍 < 1（T116 里 status 是 1.12）。
#      ②成立但①不成立 ⇒ 平坦方向是真缺陷但不解释 IC 劣化，据实记录。
#   Q2「修好的门控能否超过纯加性」：T119 vs **T115**，仍按 IC 优先协议
#      （4/4 且 Δ中位 ≥0.005，跌日分层否决门优先）。预期不过。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-18-idxrel"

"$PY" -u scripts/exp/exp_nam_gate.py \
  --stocks 6000 --years 13 --end 2022-09-05 --target returns \
  --gate-mode scalar --gate-scalar-col macro_m1m2_gap --group-norm \
  --skip-baseline --allow-degenerate-downside-risk --cache-dir "$CACHE" \
  --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
  --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
  --y-scale 2 --expert-hidden 16 --lr 2e-3 \
  --store-dtype fp16 --seeds 42,11,23,37 \
  --save-model-dir "models/nam_gate/T119_groupnorm" \
  --plot-dir "diagnose_output/nam_gate_T119" \
  --output "diagnose_output/T119_groupnorm.json"
rc=$?
echo "[T119] exit=$rc  $(date '+%H:%M:%S')"
exit $rc
