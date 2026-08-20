#!/usr/bin/env bash
# T125：门控轴最后一个未测组合 —— **数据驱动簇 + 族输出归一化同时开**。
#
# 立论（唯一有分量的那条：两个干预互补而非重复）
#   T119 单开归一化没救回来，可能是因为它作用在一个**已经互相抵消的混合物**上：
#     手工 12 族内聚 −0.0004（与随机划分不可区分，T117），族内成员符号混杂，
#     S_k 里信号早被抵消，把这么个东西标准化，锁死幅度也没有意义。
#   T118 单换簇也没救回来，可能是因为簇虽然内聚 0.35~0.50、信号方向一致，
#     但幅度自由，门于是把预算花在**最小簇的尺度**上（实测 c3 n=7 独占权重 2.14，
#     spearman(门前簇输出量级, 门控权重) = −0.943）。
#   合并是唯一同时具备「族信号方向一致」+「幅度不可被专家吸收」的格子。
#
# ⚠ 先验仍然是低：T119 已实测门控敏感度向量 `a` 在开/关归一化时逐种子余弦
#   +0.9908~+1.0000、‖a‖ 同量级 ⇒ `a` 的落点对「幅度自由度是否存在」不敏感。
#   但这个格子确实没测过，且它是本轴最后一个未测组合，跑完这条线才算真正封口。
#
# 变量隔离：与 T116 的差别恰好是 --group-map-file + --group-norm 两项同时开
#   （T118 只开前者、T119 只开后者）。面板同为 224 列、门控同为 scalar +
#   macro_m1m2_gap、超参与 dtype 全同、四种子同号。
#   ⚠ 窗口必须沿用 --years 13 --end 2022-09-05：要与 T115/T118/T119 配对判定，
#     换窗就换考卷（[[holdout-ic-not-comparable-across-windows]]）。
#     该窗 8.98M 样本 / fp16 3.75 GiB，走 cuda 驻留，不需要 --store-device host。
#
# ── 预注册判定（出结果前写死）────────────────────────────────────────────────
#   Q1「是否真互补」：必须**同时**赢过两个亲本 —— T125 vs T118 且 T125 vs T119，
#      两者都要跌日 ΔIC ≥3/4 为正且整体 Δ 中位 > 0。
#      只赢一个 ⇒ 不是互补，是其中一个变量单独在起作用，据实记录。
#   Q2「能否超过纯加性」：T125 vs T115，IC 优先协议（4/4 且 Δ中位 ≥0.005），
#      跌日分层否决门优先。预期不过。
#   Q3「机制是否变了」（信息量最大的一条）：
#      T118 实测 spearman(门前簇输出量级, 门控权重) = −0.943、门把 2.14 的权重
#      堆在 n=7 的最小簇 c3 上。若「互补」成立，本臂应看到该 spearman **塌向 0**、
#      且门不再独占 c3。**机制变了但 IC 没变 ⇒ 记为「缺陷是真的但不解释 IC」**，
#      这仍是有价值的结论，不许事后改判成晋级。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-18-idxrel"
MAP="scripts/exp/t118_group_map.json"

[ -f "$MAP" ] || { echo "[ABORT] 缺簇映射 $MAP，先跑:"; \
  echo "  $PY -u scripts/exp/diag_factor_clustering.py --emit-group-map $MAP"; exit 2; }

"$PY" -u scripts/exp/exp_nam_gate.py \
  --stocks 6000 --years 13 --end 2022-09-05 --target returns \
  --gate-mode scalar --gate-scalar-col macro_m1m2_gap \
  --group-map-file "$MAP" --group-norm \
  --skip-baseline --allow-degenerate-downside-risk --cache-dir "$CACHE" \
  --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
  --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
  --y-scale 2 --expert-hidden 16 --lr 2e-3 \
  --store-dtype fp16 --seeds 42,11,23,37 \
  --save-model-dir "models/nam_gate/T125_clust_gnorm" \
  --plot-dir "diagnose_output/nam_gate_T125" \
  --output "diagnose_output/T125_clust_gnorm.json"
rc=$?
echo "[T125] exit=$rc  $(date '+%H:%M:%S')"
exit $rc
