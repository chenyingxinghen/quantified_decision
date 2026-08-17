#!/usr/bin/env bash
# T092 第二段：NAM / XGBoost / LightGBM 的**回测**对照（IC 对照见 exp_tree_vs_nam.py）。
#
# 训练截止统一 2022-09-05，回测窗口用两段已有先例的 OOS：
#   熊 2022-09-05 → 2024-08-05，牛 2024-08-05 → 2026-08-05
# 三个模型的训练/验证结构一致：树走 TrainingConfig.TRAIN_TEST_SPLIT=0.8，
# NAM 走 --folds 0.8:1.0，都是「前 80% 训练、后 20% 早停选型」。
#
# ⚠ 与 IC 对照的口径差异（有意为之）：这里的树是**生产配置**（train_model.py 的全部特征），
# 而 exp_tree_vs_nam.py 为了单变量可比，把树限制在 NAM 的特征集（剔手工交互 + 剔 forecast 族）。
# 前者回答「实盘该用哪个」，后者回答「同样输入下哪个模型更强」。
#
# ⚠ 回测 stdout 必须 UTF-8，否则会在 5 分钟预加载之后才崩（见 [[backtest-stdout-gbk-trap]]）。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
TRAIN_END="2022-09-05"
W1_S="2022-09-05"; W1_E="2024-08-05"   # 熊
W2_S="2024-08-05"; W2_E="2026-08-05"   # 牛
LOG=diagnose_output/T092_bt_driver.log

say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ── 1. 生产配置训练两棵树 ───────────────────────────────────────────────
if [ ! -f diagnose_output/.t092_trees_trained ]; then
  say "训练 xgboost + lightgbm（生产配置，截止 $TRAIN_END）"
  "$PY" -u scripts/train_model.py --end "$TRAIN_END" --stocks 800 \
      --models xgboost lightgbm --cache-dir "$CACHE" \
      --skip-cache-update --no-update-latest \
      > diagnose_output/T092_prod_trees_train.log 2>&1
  rc=$?
  [ $rc -ne 0 ] && { say "树训练失败 rc=$rc"; tail -25 diagnose_output/T092_prod_trees_train.log; exit 1; }
  touch diagnose_output/.t092_trees_trained
fi

# `--models xgboost lightgbm` 会把两个 pkl 写进**同一个** xl_* 目录，不是各自一个目录。
# 传目录给 run_backtest 只会加载 mtime 最新的那个 pkl（见 ml_factor_strategy._load_smart_model），
# 所以必须显式指到文件；早先按 `ls -dt models/xg_*` 取目录会抓到几周前绑旧缓存的存档。
PROD_DIR=$(ls -dt models/xl_* 2>/dev/null | head -1)
XGB_DIR="$PROD_DIR/xgboost_factor_model.pkl"
LGB_DIR="$PROD_DIR/lightgbm_factor_model.pkl"
[ -f "$XGB_DIR" ] && [ -f "$LGB_DIR" ] || { say "生产模型缺失于 $PROD_DIR"; exit 1; }
say "xgb=$XGB_DIR  lgb=$LGB_DIR"

# ── 2. NAM 存档（同折结构 0.8:1.0）─────────────────────────────────────
NAM_DIR=models/nam_gate/T092_nam_s42
if [ ! -f "$NAM_DIR/nam_gate_factor_model.pkl" ]; then
  say "训练 NAM 并存档 → $NAM_DIR"
  "$PY" -u scripts/exp/exp_nam_gate.py --stocks 800 --years 13 --end "$TRAIN_END" \
      --disable-gate --target returns --y-scale 2 --lambda-lb 0 --epochs 60 --min-epochs 20 \
      --skip-baseline --allow-degenerate-downside-risk --cache-dir "$CACHE" \
      --folds 0.8:1.0 --seed 42 --drop-groups forecast \
      --save-model-dir "$NAM_DIR" --output diagnose_output/T092_nam_bt_s42.json \
      > diagnose_output/T092_nam_bt_train.log 2>&1
  rc=$?
  [ $rc -ne 0 ] && { say "NAM 训练失败 rc=$rc"; tail -25 diagnose_output/T092_nam_bt_train.log; exit 1; }
fi

# ── 3. 回测：3 个单模型 + xgb/lgb 等权集成，各 2 个窗口 ────────────────
run_bt () {  # run_bt <标签> <窗口起> <窗口止> <额外参数...>
  local tag="$1" s="$2" e="$3"; shift 3
  local out="diagnose_output/T092_bt_${tag}_${s}_train.log"
  [ -f "diagnose_output/.bt_${tag}_${s}.done" ] && { say "跳过 $tag $s（已完成）"; return; }
  say "回测 $tag  $s → $e"
  "$PY" -u scripts/run_backtest.py --start "$s" --end "$e" --cache-dir "$CACHE" \
      --tag "T092_${tag}" "$@" > "$out" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "回测失败 $tag $s rc=$rc"; tail -20 "$out";
  else touch "diagnose_output/.bt_${tag}_${s}.done"; fi
}

for W in "$W1_S $W1_E" "$W2_S $W2_E"; do
  set -- $W
  run_bt xgb "$1" "$2" --model "$XGB_DIR"
  run_bt lgb "$1" "$2" --model "$LGB_DIR"
  run_bt nam "$1" "$2" --model "$NAM_DIR"
  run_bt ens "$1" "$2" --model "$XGB_DIR" --ensemble-model "$LGB_DIR"
done

say "全部完成"
