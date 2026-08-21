#!/usr/bin/env bash
# T133（②）：**头部集中的损失** —— 剂量-反应扫描 y_scale ∈ {1,2,4,8} × 2 种子。
#
# 立论
#   T130 量到两件事：
#     ① 头部（前 200 候选池内重排取前 20）的**诚实折外天花板**是 top20 超额 +0.0176，
#        生产件只拿到 +0.0071 —— **2.5 倍空间**，而且这个空间是"重排族权重"就能吃到的。
#     ② 全截面 rank IC **看不见**头部改动（同一个重排在全截面上只有 +0.0011 / t=2.13，
#        persist 类甚至转负）。见 [[head-ruler-is-blind-in-full-cross-section-ic]]。
#   而现在的训练配方**两头都对着全截面**：ListNet 在全截面上算损失，早停用全截面 rank IC
#   选 checkpoint。**我们在优化一个不交易的目标。**
#
#   `--y-scale` 正好是这条轴的旋钮：ListNet 的目标是 `softmax(y * y_scale)`，
#   y_scale 越大，目标质量越集中在当日排名的**头部**。冠军 y2 是 [[nam-hpo-axes-closed]]
#   扫出来的 —— **但那轮是用全截面 IC 判的，也就是那把瞎尺子**。所以这不是重开已关的轴，
#   是**用能看见头部的尺子重判一条以前判错过的轴**。
#
# ⚠ 必须诚实说明的先验
#   y_scale 变大 ⇒ 有效样本量变小（softmax 目标趋近 one-hot，每天只有几只票贡献梯度），
#   所以既可能"对准了头部"也可能"只是噪声更大"。真实响应很可能是**单峰**而非单调。
#
# ── 设计：4 个剂量点 × 2 种子（而不是 2 个点 × 4 种子）────────────────────────
#   总预算一样（8 次种子级训练），但换来的是**剂量-反应曲线**。
#   代价说清楚：每臂只有 2 种子，原先「4/4 为正」那条门**用不了**
#   （2/2 为正在纯噪声下就有 25% 概率）。补偿手段是判**跨剂量的趋势一致性**，
#   见下面的判据 —— 一条 8 个点都对得上的单峰/单调曲线，比 2 个点各 4 种子更难被噪声伪造。
#
#   **y1 是证伪臂**：若头部集中机制为真，y1 应当**劣于** y2。
#   若 y1 反而也赢 y2，那赢的就不是"对准头部"，而是"动一下就有"的种子噪声 —— 直接关轴。
#
#   **y2 重跑而不复用 T115 存档**：命令行与 run_t115_idxrel.sh 逐字相同（只差 y_scale
#   与存档名），但 T115 那批是**旧代码**训的（此后合过复权口径与门控中心化修复）。
#   同批重训才能保证四臂唯一的差异是 y_scale。附带好处：新 y2 与 T115_s42/s11 的
#   差值就是代码漂移的直接读数，对不上说明有东西变了。
#
# ── 预注册判定（出结果前写死）───────────────────────────────────────────────
#   主判据（头部尺子，T130 口径）：逐日 top-20 超额，候选池 = 前 200，
#     **逐种子配对** vs 同种子的 y2 臂（s42 对 s42，s11 对 s11）。晋级需三条同时成立：
#       (a) 该臂 **2/2 种子** Δ 为正；
#       (b) Δ 中位 ≥ **+0.002**（头部超额比 IC 小一个数量级，0.005 那条线是给全截面
#           IC 定的，直接搬过来不合口径；0.002 约等于 T130 里 `global` 臂 +0.0042 的一半，
#           取"能被现有工具稳定分辨"的量级）；
#       (c) **趋势一致**：两个种子给出的四臂排序方向不冲突，且曲线形状是单调或单峰
#           （y1<y2<y4<y8、y1<y2>y4>y8 之类），不是锯齿。
#     (c) 不成立 ⇒ 无论 (a)(b) 多好看都只记录不晋级 —— 锯齿是噪声的形状。
#   否决门（优先于主判据）：**跌日** Δ 必须 2/2 为正。
#     T130 里 `global`/`ifelse` 正是头部赢、跌日 2/4 而被拒的，同一把尺子同一条门。
#   证伪读数（不是判据，是机制检验）：y1 vs y2。y1 若不劣于 y2 ⇒ 机制不成立，整轴关闭。
#   副判据（记录用，不参与晋级）：全截面 holdout rank IC。
#     **预期它会掉** —— 若它掉而头部涨，那正是"两把尺子会给相反结论"的又一个例证；
#     若两把尺子同时涨，反而要怀疑是不是种子运气。
#
#   ⚠ 判定脚本：scripts/exp/diag_seed_ensemble.py 已经同时报两把尺子 + 涨跌日分层，
#     对 {T133_yscale1_s*, T133_yscale2_s*, T133_yscale4_s*, T133_yscale8_s*} 各跑一遍。
#
# 窗口必须沿用 --years 13 --end 2022-09-05：要与 T115 配对，换窗就换考卷
#   （[[holdout-ic-not-comparable-across-windows]]）。该窗 8.98M 样本 / fp16 3.75 GiB，
#   走 cuda 驻留，不需要 --store-device host。
#
# 排期：4 次调用各自重建数据集（--y-scale 是标量，塞不进一次调用），
#   每次 ≈ 20min 准备 + 2×50min 训练 + 2×8min 存档诊断 ≈ 2.3h，四臂合计 ≈ 9h。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-18-idxrel"

[ -f "$CACHE/factor_cache_manifest.json" ] || {
  echo "[ABORT] 缺缓存清单 $CACHE/factor_cache_manifest.json"; exit 2; }

rc_all=0
for YS in 1 2 4 8; do
  echo "===== T133 y_scale=$YS  $(date '+%F %H:%M:%S') ====="
  "$PY" -u scripts/exp/exp_nam_gate.py \
    --stocks 6000 --years 13 --end 2022-09-05 --disable-gate --target returns \
    --skip-baseline --allow-degenerate-downside-risk --cache-dir "$CACHE" \
    --folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
    --epochs 60 --min-epochs 20 --lambda-lb 0 --chunk-days 4 \
    --y-scale "$YS" --expert-hidden 16 --lr 2e-3 \
    --store-dtype fp16 --seeds 42,11 \
    --save-model-dir "models/nam_gate/T133_yscale${YS}" \
    --plot-dir "diagnose_output/nam_gate_T133_y${YS}" \
    --output "diagnose_output/T133_yscale${YS}.json"
  rc=$?
  echo "[T133 y=$YS] exit=$rc  $(date '+%F %H:%M:%S')"
  [ $rc -ne 0 ] && rc_all=$rc
done
echo "[T133] all done exit=$rc_all  $(date '+%F %H:%M:%S')"
exit $rc_all
