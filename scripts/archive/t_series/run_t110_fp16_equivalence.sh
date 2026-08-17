#!/usr/bin/env bash
# T110：先验 fp16 驻留的**数值等价性**，再用它补完 T108 的 s23/s37。
#
# 背景（T109）：全量池 Xtr+Xva = 7.39 GiB > 6.00 GiB 显存，Windows 驱动的
# sysmem fallback 静默把溢出页到主机内存，每个 day-step 走 PCIe。fp16 驻留
# （计算仍 fp32）降到 3.70 GiB，实测 383.4s/epoch → 30.7s/epoch = **12.5x**。
#
# 但这是数值改动，不能直接混进 n=4 判决。所以第一步是等价性检验：
#   用**已经跑完的 s11**（fp32 holdout 0.09818）同种子同配置重跑一次 fp16，
#   若 |Δ IC| < 0.002（≈ 1/3 配对 MDE）则认定 fp16 是噪声级扰动，
#   s23/s37 可以用 fp16 跑并进 T108 判决（判决靠回测，MDE 30~50pp，
#   fp16 的 1e-3 相对输入扰动远在其下）。
#   若 |Δ IC| ≥ 0.002 则**放弃混用**，s23/s37 回退 fp32（9.2h），
#   fp16 只用于今后的新实验。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
CACHE="database/system_data/factors_cache_2026-08-14-fwdadjust"
COMMON="--stocks 6000 --years 13 --end 2022-09-05 --target returns --skip-baseline \
--allow-degenerate-downside-risk --cache-dir $CACHE \
--folds 0.8:1.0 --drop-groups forecast --select-holdout 0.4 \
--epochs 60 --min-epochs 20 --y-scale 2 --expert-hidden 16 --lr 2e-3 \
--gate-mode softmax --lambda-lb 0.01"

train () {  # train <seed> <json> <modeldir> <plotdir>
  echo "[TRAIN] s$1 fp16  $(date '+%H:%M:%S')"
  "$PY" -u scripts/exp/exp_nam_gate.py $COMMON --seed "$1" --store-dtype fp16 \
    --save-model-dir "$3" --plot-dir "$4" --output "$2" \
    > "${2%.json}.log" 2>&1
  local rc=$?
  echo "[$([ $rc -eq 0 ] && echo OK || echo FAIL)] s$1 rc=$rc  $(date '+%H:%M:%S')"
  return $rc
}

# ── 第一步：等价性检验（s11 已有 fp32 读数 0.09818）────────────────────
J11="diagnose_output/T110_fp16_s11.json"
[ -f "$J11" ] || train 11 "$J11" "models/nam_gate/T110_fp16_s11" \
                            "diagnose_output/nam_gate_T110_fp16_s11"
VERDICT=$("$PY" - <<'PYEOF'
import json
def ic(p):
    d = json.load(open(p, encoding='utf-8'))['folds']
    r = [v['nam_gate'] for v in d.values() if isinstance(v, dict) and 'nam_gate' in v]
    return r[0]['rank_ic_holdout']
try:
    a, b = ic('diagnose_output/T106_lb001_s11.json'), ic('diagnose_output/T110_fp16_s11.json')
    print(f"{'PASS' if abs(a-b) < 0.002 else 'FAIL'} fp32={a:.5f} fp16={b:.5f} delta={b-a:+.5f}")
except Exception as e:
    print(f"ERROR {e}")
PYEOF
)
echo "[等价性] $VERDICT"
echo "$VERDICT" > diagnose_output/T110_equivalence.txt
case "$VERDICT" in PASS*) DT=fp16 ;; *) DT=fp32 ;; esac
echo "[决定] s23/s37 使用 --store-dtype $DT"

# ── 第二步：补完 T108 的 s23/s37 + 回测 ────────────────────────────────
for S in 23 37; do
  J="diagnose_output/T106_lb001_s${S}.json"
  if [ ! -f "$J" ]; then
    echo "[TRAIN] s$S $DT  $(date '+%H:%M:%S')"
    "$PY" -u scripts/exp/exp_nam_gate.py $COMMON --seed "$S" --store-dtype "$DT" \
      --save-model-dir "models/nam_gate/T106_lb001_s${S}" \
      --plot-dir "diagnose_output/nam_gate_T106_lb001_s${S}" \
      --output "$J" > "diagnose_output/T106_lb001_s${S}.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] s$S rc=$rc"; tail -12 "diagnose_output/T106_lb001_s${S}.log"; continue; }
    echo "[OK] train s$S  $(date '+%H:%M:%S')"
  fi
  M="models/nam_gate/T106_lb001_s${S}/nam_gate_factor_model.pkl"
  [ -f "$M" ] || { echo "[SKIP] 缺存档 s$S"; continue; }
  for W in "2022-09-05 2024-08-05" "2024-08-05 2026-08-05"; do
    set -- $W
    mk="diagnose_output/.bt108_s${S}_${1}.done"
    [ -f "$mk" ] && continue
    echo "[$(date '+%H:%M:%S')] 回测 gate s$S $1"
    "$PY" -u scripts/run_backtest.py --start "$1" --end "$2" --cache-dir "$CACHE" \
        --tag "T108_gate_s${S}" --model "$M" \
        > "diagnose_output/T108_bt_gate_s${S}_${1}.log" 2>&1
    [ $? -eq 0 ] && touch "$mk"
  done
done
echo "[DONE] $(date '+%H:%M:%S')"
