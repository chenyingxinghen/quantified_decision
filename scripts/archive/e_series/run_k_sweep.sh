#!/usr/bin/env bash
# E4: 持仓档位 K 扫描 {20(已有), 40, 60}
#
# 依据：IR ≈ IC·√N。当前 IC≈0.097 属弱信号，且 T037 已测出 Top-1/5/20 三档
# 头部分位都在 0.58–0.62（头部是平的）。这意味着 K 从 20 扩到 40/60 几乎不
# 损失单只信号质量，却按 √K 削减特异性风险。弱信号 + 平头部的最优 K 应明显 > 20。
#
# 反证（须诚实标注）：T038 里 XGBoost 的 Top-5(-7.99%) 优于 Top-20(-17.21%)。
# 但那是另一个模型，且两者都是负的，证据力弱。
#
# E1b 之后的口径修正（重要）
# --------------------------
# 原版只在 T045(seed42) 上扫 K。E1b 已证明 seed42 是种子彩票的极值抽样
# （牛市 96.5% = 10 种子最大值），单种子结论不可信。K 虽然不涉及重训练、
# 同种子内是配对比较，但「K=40 更好」这一结论能否跨种子成立必须验证。
# 因此这里在 σ_seed 种子池的 4 个种子 {42,11,23,37} 上各扫一遍，
# 判定用配对差 (K=40 或 60) − (K=20)，n=4 对。
#
# 用法: bash scripts/archive/e_series/run_k_sweep.sh
set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1

# 2026-08-12：必须显式设 UTF-8。stdout 重定向到文件时 Windows Python 用 gbk 编码，
# ml_factor_strategy.py:203 的 "✓" 会抛 UnicodeEncodeError——而且崩在 ~290 s 的
# 因子预加载之后，每次白烧 5 分钟。这是驱动层修复，不动回测代码（E4 期间禁改）。
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# --- 单实例锁 + 训练互斥 -----------------------------------------------------
LOCK="diagnose_output/.k_sweep.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then
    echo "[LOCK] 已有 K sweep 在跑 (pid=$old)，本次退出。若确认已死: rm $LOCK"
    exit 1
  fi
  echo "[LOCK] 发现陈旧锁 (pid=$old 已不存在)，接管"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

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
  echo "[BLOCK] 检测到 exp_nam_gate.py 仍在训练，拒绝并发回测。"
  exit 1
fi

# seed -> 模型目录（同配置，仅训练种子不同）
model_dir_of () {
  case "$1" in
    42) echo "models/nam_gate/T045" ;;
    *)  echo "models/nam_gate/T068_seed$1" ;;
  esac
}
name_of () {
  case "$1" in
    42) echo "T045" ;;
    *)  echo "T068_seed$1" ;;
  esac
}

for SEED in 42 11 23 37; do
  MODEL=$(model_dir_of "$SEED")
  NAME=$(name_of "$SEED")
  if [ ! -f "${MODEL}/nam_gate_factor_model.pkl" ]; then
    echo "[SKIP] ${MODEL} 缺失，跳过 seed=${SEED}"
    continue
  fi
  # 2026-08-12 缩减：只跑 K=40。理由见台账 E4 节——s42 上已有 k20/k40/k60 三点，
  # excess 单调衰减 (bull 36.9→18.5→15.0)，K=60 已被 K=40 支配，再跑是无用功。
  # 只需把 K=40 vs K=20 的配对差补到 n=4 种子即可判定 E4。
  for K in 40; do
    for WIN in bull bear; do
      if [ "$WIN" = "bull" ]; then S=2024-08-05; E=2026-08-05; else S=2022-09-05; E=2024-08-05; fi
      TAG="T069_s${SEED}_k${K}_${WIN}"
      OUT="backtest_result/${NAME}/nam/conf0_${TAG}_minp1_nost_mp${K}_${S}_to_${E}"
      TRADES="${OUT}/backtest_trades.csv"

      if [ ! -f "$TRADES" ]; then
        echo "[BT ] seed=${SEED} K=${K} ${WIN}"
        # 必须 --exclude-st（触发目录名 nost 段），与 T045 基线同口径。
        # 旧版这里误写 --include-st，会产出无 nost 段的目录，不可比。
        "$PY" scripts/run_backtest.py \
          --model "$MODEL" --start "$S" --end "$E" \
          --min-confidence 0 --risk-min-price 1 --exclude-st \
          --max-positions "$K" --tag "$TAG" \
          > "diagnose_output/${TAG}_backtest.log" 2>&1
      else
        echo "[SKIP BT ] $TRADES 已存在"
      fi

      if [ ! -f "$TRADES" ]; then
        actual=$(ls -d backtest_result/${NAME}/nam/*${TAG}* 2>/dev/null | head -1)
        if [ -n "$actual" ]; then
          echo "[口径错配] 期望 $OUT"
          echo "           实得 $actual -> 拒绝纳入判定"
        else
          echo "[FAIL] $TRADES 未产出，见 diagnose_output/${TAG}_backtest.log"
        fi
        continue
      fi

      if [ ! -f "diagnose_output/random_null_${TAG}.json" ]; then
        echo "[NULL] ${TAG}"
        "$PY" scripts/exp/diag_random_null.py \
          --trades "$TRADES" --n-sims 2000 \
          --buy-cost 0.0008 --sell-cost 0.0013 \
          --out "diagnose_output/random_null_${TAG}.json" \
          > "diagnose_output/${TAG}_random.log" 2>&1
      else
        echo "[SKIP NULL] random_null_${TAG}.json 已存在"
      fi
    done
  done
done

echo "=== K SWEEP DONE ==="
