#!/usr/bin/env bash
# E1b: 十种子零效应分布驱动脚本
# 对每个同配置种子跑双窗口真实成本回测 + 随机零假设，产出 σ_seed 所需样本。
# 用法: bash scripts/archive/e_series/run_seed_null_sweep.sh
set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
cd "$ROOT" || exit 1

# --- 单实例锁 ---------------------------------------------------------------
# 踩过的坑：TaskStop 只杀 bash 外壳，python 子进程会继续跑；多次重启后曾出现
# 三条 sweep 并发（其中一条还是旧的 --include-st 错误口径），互相抢内存导致
# 进程被静默 kill，且日志互相覆盖。此锁保证任何时刻只有一条 sweep 在跑。
LOCK="diagnose_output/.seed_null_sweep.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null)
  if ps -p "$old" >/dev/null 2>&1; then
    echo "[LOCK] 已有 sweep 在跑 (pid=$old)，本次退出。若确认已死: rm $LOCK"
    exit 1
  fi
  echo "[LOCK] 发现陈旧锁 (pid=$old 已不存在)，接管"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
# 启动前清场：杀掉任何遗留的 run_backtest / diag_random_null 进程
"$PY" - <<'PYEOF'
import os, subprocess, sys
try:
    out = subprocess.run(
        ['wmic', 'process', 'where', "name='python.exe'", 'get', 'ProcessId,CommandLine', '/format:csv'],
        capture_output=True, text=True, timeout=30).stdout
except Exception:
    out = ''
me = os.getpid()
killed = []
for line in out.splitlines():
    if 'run_backtest.py' in line or 'diag_random_null.py' in line:
        pid = line.strip().rsplit(',', 1)[-1]
        if pid.isdigit() and int(pid) != me:
            subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
            killed.append(pid)
if killed:
    print('[LOCK] 已清理遗留回测进程:', ' '.join(killed))
PYEOF

BULL_START=2024-08-05; BULL_END=2026-08-05
BEAR_START=2022-09-05; BEAR_END=2024-08-05

run_one () {
  local model_dir="$1"   # models/nam_gate/XXX
  local name="$2"        # 结果目录名 XXX
  local win="$3"         # bull | bear
  local tag="$4"

  local start end
  if [ "$win" = "bull" ]; then start=$BULL_START; end=$BULL_END; else start=$BEAR_START; end=$BEAR_END; fi

  local outdir="backtest_result/${name}/nam/conf0_${tag}_minp1_nost_mp20_${start}_to_${end}"
  local trades="${outdir}/backtest_trades.csv"
  local nulljson="diagnose_output/random_null_${tag}.json"

  if [ ! -f "$trades" ]; then
    echo "[BT ] $name $win -> $tag"
    # 注意：目录名里的 "nost" 段由 --exclude-st 触发（run_backtest.py:256），
    # 语义是「ST 已被排除」。T045 基线正是 nost 口径，因此这里必须用
    # --exclude-st；误用 --include-st 会产出不含 nost 段的目录，
    # 与基线口径不可比。
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
    # 口径守卫：若回测确实跑完但落到了别的目录名，说明命令行口径与基线不一致
    local actual
    actual=$(ls -d backtest_result/${name}/nam/*${tag}* 2>/dev/null | head -1)
    if [ -n "$actual" ]; then
      echo "[口径错配] 期望 $outdir"
      echo "           实得 $actual"
      echo "           -> 命令行开关与 T045 基线不一致，拒绝纳入 σ_seed"
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

# --- 6 个 T068 新种子：双窗口 ---
for s in 11 23 37 53 67 89; do
  run_one "models/nam_gate/T068_seed${s}" "T068_seed${s}" bull "T068_s${s}_bull"
  run_one "models/nam_gate/T068_seed${s}" "T068_seed${s}" bear "T068_s${s}_bear"
done

# --- 3 个 T047 老种子：补熊市窗口 ---
for s in 7 123 2024; do
  run_one "models/nam_gate/T047_s${s}" "T047_s${s}" bear "T047_s${s}_bear"
done

echo "=== SWEEP DONE ==="
