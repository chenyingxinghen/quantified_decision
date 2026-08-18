#!/usr/bin/env bash
# T116：门控终检 —— 外生宏观标量门（M1-M2 剪刀差）× 4 种子。
#
# 立论（2026-08-18 数据缺口讨论定案）：
#   门控轴 T108/T111 已按跌日 ΔIC 4/4 为负终审关闭，但当时 29 维 regime 全部
#   内生（池内广度/趋势/波动/涨停流），「外生宏观信号能否救门控」严格说没测过。
#   本轮用**最小容量**补上这最后一块：scalar 门只有 K=11 个参数（E6 双重中心化
#   bug 已修，见 nam_gate_model.py RegimeGate 注释），信号 = macro_m1m2_gap
#   （M1-M2 同比剪刀差，statMonth 次月 15 日 PIT 可见，滚动分位归一化）。
#
# 预注册判据（这是否决检验，不是晋级检验）：
#   门A（主判，regime-stratified）：跌日 ΔIC(T116−T115) ≤1/3 种子为正 → 关轴。
#   门B：整体配对 Δ 4/4 为负且超 MDE → 关轴（与门A 任一触发即关）。
#   两门都没触发 → 也只算「未否决」，晋级还需 Δ≥0.005 且 4/4（IC 优先协议），
#   预期概率极低。无论结果如何，此后门控轴不再重开：
#   内生 regime（T108/T111）+ 外生宏观（本轮）都已测过。
#
# 面板 = T115 的 224 列（219 + 5 列指数相对，族 K=12）。
#   2026-08-18 用户决定：T115 虽未过预注册晋级门（3/4、Δ均值 < MDE），但
#   `index_rel` 族 standalone IC 0.064~0.070 四种子一致、IR 0.46~0.53，
#   族层面信息为真（模型层面测不出是因为 5 列只占 1.55% 贡献权重），
#   据此判断晋级为新基线。故本轮配对基线是 **T115_idxrel** 而非 T113。
#   T114 瘦身臂已否决（方向为负），所以**不带** --drop-features-file。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-18-idxrel"

"$PY" -u scripts/exp/exp_nam_gate.py \
  --stocks 6000 --years 13 --end 2022-09-05 --target returns \
  --gate-mode scalar --gate-scalar-col macro_m1m2_gap \
  --skip-baseline --allow-degenerate-downside-risk --cache-dir "$CACHE" \
  --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
  --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
  --y-scale 2 --expert-hidden 16 --lr 2e-3 \
  --store-dtype fp16 --seeds 42,11,23,37 \
  --save-model-dir "models/nam_gate/T116_macro_gate" \
  --plot-dir "diagnose_output/nam_gate_T116" \
  --output "diagnose_output/T116_macro_gate.json"
rc=$?
echo "[T116] exit=$rc  $(date '+%H:%M:%S')"
exit $rc
