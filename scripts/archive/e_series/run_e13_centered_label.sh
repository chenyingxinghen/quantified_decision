#!/usr/bin/env bash
# T083 / E13：**以 7 天为中心**的多持有期标签 —— 把「降噪」和「拉长持有期」拆开
#
# E11（5/10/20）混了两个效应：① 多期平均降标签噪声；② 平均持有期从 7 天拉到 11.7 天。
# T082 证明它卖掉的是下跌日精度（下跌日 ΔIC 2/12 正）——典型的趋势倾斜，来自 ②。
# 那 ① 本身还有没有用？本轮用 5,6,7,8,9：平均持有期正好 7.0（与执行口径一致）、
# 5 个持有期仍能压噪声，禁运期只需砍 9−7=2 个交易日（E11 要砍 13 天）。
#
#   ① 有效 ⇒ ΔIC 为正且下跌日不劣化 ⇒ 这才是真 alpha，值得花回测确认。
#   ① 无效 ⇒ ΔIC≈0 ⇒ E11 的全部增益都来自趋势倾斜，标签噪声轴就此关闭。
#
# 两臂都用新代码跑：基线也必须重跑，因为 T078_mf_base 没有 T082 加的 regime 段，
# 分层门槛无从比较。这 4 次基线训练是一次性投入，之后每个轴都能复用。
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOCK="diagnose_output/.e13.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then echo "[LOCK] 已有实例 (pid=$old)"; exit 1; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

FOLDS="0.6:0.8,0.7:0.9,0.8:1.0"
COMMON="--stocks 800 --years 13 --end 2022-09-05 --disable-gate --target returns
        --y-scale 2 --lambda-lb 0 --epochs 60 --min-epochs 20 --skip-baseline
        --allow-degenerate-downside-risk"

run_arm () {
  TAG="$1"; shift
  for SEED in 42 11 23 37; do
    JSON="diagnose_output/${TAG}_s${SEED}.json"
    [ -f "$JSON" ] && { echo "[SKIP] $JSON"; continue; }
    echo "[TRAIN] $TAG seed=$SEED"
    "$PY" scripts/exp/exp_nam_gate.py $COMMON --folds "$FOLDS" --seed "$SEED" "$@" \
      --output "$JSON" > "diagnose_output/${TAG}_s${SEED}_train.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] $TAG seed=$SEED rc=$rc"; tail -20 "diagnose_output/${TAG}_s${SEED}_train.log"; }
  done
}

# 基线臂（带 regime 段），再跑改动臂 —— 串行，两个训练同时跑会把内存打爆（08-08 事故）
run_arm T083_base
run_arm T083_mh5to9 --multi-horizon 5,6,7,8,9

echo "=== T083 E13 DONE ==="
"$PY" scripts/exp/analyze_multifold.py --prefix T083_mh5to9 --base-prefix T083_base || true
