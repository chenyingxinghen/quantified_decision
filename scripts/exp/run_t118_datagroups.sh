#!/usr/bin/env bash
# T118：门控轴结构性修复①——**用数据驱动簇替换手工分族**，其余一切不变。
#
# 立论（T117，零训练测量）：
#   手工 12 族的内聚度（族内成员日度贡献 IC 序列的平均两两相关）是 −0.0004，
#   随机 12 分组是 −0.0016 ± 0.0048 ⇒ **z = +0.2，统计上不可区分**。
#   而数据驱动聚类内聚 +0.3475（z = +73.0）。相关矩阵 top12 主成分吃掉 75.2% 方差
#   ⇒ 224 列的预测行为只有约 3~12 个自由度。
#   门只能看到族内求和 S_k，族内符号混杂时 S_k 里已互相抵消 ⇒ 门连"该放大谁"
#   都收不到。这是 T116 门控只能学出静态解的**第一个**候选原因。
#
# 变量隔离：与 T116 的**唯一**差别是 --group-map-file（分族来源）。
#   面板同为 224 列、门控同为 scalar + macro_m1m2_gap、超参与 dtype 全同。
#   簇映射只由**训练段**导出（diag_factor_clustering.py --emit-group-map），
#   拿验证段聚类等于把验证信息带进架构选择。
#   12 簇经小簇合并（min-cluster-size 5）后为 6 簇 —— 单例簇会复刻 status 病灶。
#
# 两个判定，分别回答两个问题（预注册）：
#   Q1「分族是不是 T116 失败的原因」：T118 vs **T116**（同为门控，只差基）。
#      跌日 ΔIC ≥3/4 为正且整体 Δ 中位 > 0 ⇒ 分族确实是主因之一，
#      门控这条线值得继续修（下一步是族输出归一化 / 残差门控）。
#      跌日 ≤1/3 为正 ⇒ 换基没救回来，第一个候选原因被排除。
#   Q2「修好基的门控能否超过纯加性」：T118 vs **T115**（当前基线，纯加性）。
#      仍按 IC 优先协议：4/4 且 Δ中位 ≥0.005 才算晋级；跌日分层否决门优先。
#      预期不过 —— T116 反向 10× MDE，换基要补回 0.028 才够。
#
# ⚠ Q1 过、Q2 不过是**最可能**的结果，那意味着"分族是真缺陷但不是唯一缺陷"，
#   届时应继续修结构（归一化/残差）而不是宣布门控可用。
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
  --group-map-file "$MAP" \
  --skip-baseline --allow-degenerate-downside-risk --cache-dir "$CACHE" \
  --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
  --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
  --y-scale 2 --expert-hidden 16 --lr 2e-3 \
  --store-dtype fp16 --seeds 42,11,23,37 \
  --save-model-dir "models/nam_gate/T118_datagroups" \
  --plot-dir "diagnose_output/nam_gate_T118" \
  --output "diagnose_output/T118_datagroups.json"
rc=$?
echo "[T118] exit=$rc  $(date '+%H:%M:%S')"
exit $rc
