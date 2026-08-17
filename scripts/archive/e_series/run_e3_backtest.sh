#!/usr/bin/env bash
# E3: 标签残差化 (beta_size) 四种子双窗口回测 + 随机零假设
# 判定口径：与同种子 none 基线做配对比较（不看 Rank IC，残差化后 IC 必然下降）
# 用法: bash scripts/archive/e_series/run_e3_backtest.sh
set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1

# --- 单实例锁（同 run_seed_null_sweep.sh 的教训） -----------------------------
LOCK="diagnose_output/.e3_backtest.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then
    echo "[LOCK] 已有 E3 回测在跑 (pid=$old)，本次退出。若确认已死: rm $LOCK"
    exit 1
  fi
  echo "[LOCK] 发现陈旧锁 (pid=$old 已不存在)，接管"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# 训练未结束不许开跑（训练 + 回测并发会打爆内存）
if "$PY" - <<'PYEOF'
import subprocess, sys
try:
    out = subprocess.run(['wmic','process','where',"name='python.exe'",'get','CommandLine','/format:csv'],
                         capture_output=True, text=True, timeout=30).stdout
except Exception:
    out = ''
sys.exit(0 if 'exp_nam_gate.py' in out else 1)
PYEOF
then
  echo "[BLOCK] 检测到 exp_nam_gate.py 仍在训练，拒绝并发回测。等训练结束再跑。"
  exit 1
fi

BULL_START=2024-08-05; BULL_END=2026-08-05
BEAR_START=2022-09-05; BEAR_END=2024-08-05

run_one () {
  local model_dir="$1"
  local name="$2"
  local win="$3"
  local tag="$4"

  local start end
  if [ "$win" = "bull" ]; then start=$BULL_START; end=$BULL_END; else start=$BEAR_START; end=$BEAR_END; fi

  local outdir="backtest_result/${name}/nam/conf0_${tag}_minp1_nost_mp20_${start}_to_${end}"
  local trades="${outdir}/backtest_trades.csv"
  local nulljson="diagnose_output/random_null_${tag}.json"

  if [ ! -f "$trades" ]; then
    echo "[BT ] $name $win -> $tag"
    "$PY" scripts/run_backtest.py \
      --model "$model_dir" \
      --start "$start" --end "$end" \
      --min-confidence 0 --risk-min-price 1 --exclude-st \
      --max-positions 20 --tag "$tag" \
      > "diagnose_output/${tag}_backtest.log" 2>&1
  else
    echo "[SKIP BT ] $trades 已存在"
  fi

  if [ ! -f "$trades" ]; then
    local actual
    actual=$(ls -d backtest_result/${name}/nam/*${tag}* 2>/dev/null | head -1)
    if [ -n "$actual" ]; then
      echo "[口径错配] 期望 $outdir"
      echo "           实得 $actual  -> 拒绝纳入判定"
    else
      echo "[FAIL] 回测未产出 $trades ; 见 diagnose_output/${tag}_backtest.log"
    fi
    return 1
  fi

  if [ ! -f "$nulljson" ]; then
    echo "[NULL] $tag"
    "$PY" scripts/exp/diag_random_null.py \
      --trades "$trades" --n-sims 2000 \
      --buy-cost 0.0008 --sell-cost 0.0013 \
      --out "$nulljson" \
      > "diagnose_output/${tag}_random.log" 2>&1
  else
    echo "[SKIP NULL] $nulljson 已存在"
  fi
}

for s in 42 11 23 37; do
  MD="models/nam_gate/T070_e3_beta_size_s${s}"
  if [ ! -f "${MD}/nam_gate_factor_model.pkl" ]; then
    echo "[SKIP] ${MD} 模型缺失，跳过 seed=${s}"
    continue
  fi
  run_one "$MD" "T070_e3_beta_size_s${s}" bull "T070_e3_s${s}_bull"
  run_one "$MD" "T070_e3_beta_size_s${s}" bear "T070_e3_s${s}_bear"
done

echo "=== E3 BACKTEST DONE ==="
