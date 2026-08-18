#!/usr/bin/env bash
# T113~T116 串行队列驱动（2026-08-18 定案的四个批次，一条命令跑完）。
#
# 特性：
#   - 批级断点续跑：某批的最后一个种子 JSON 已存在则整批跳过（可安全重复执行）。
#   - 顺序：T113 基线 → T114 瘦身 → T116 门控终检 → [构建 idxrel 缓存] → T115 指数特征。
#     T115 放最后因为它多一步 ~1h 的 CPU 缓存构建；缓存构建放在 T116 训练之后
#     启动也可以，但串行最稳（笔记本别同时压 GPU+4 核 IO）。
#   - 预计总时长：4 批 × ~3h GPU + ~1h CPU ≈ 13h，全程风扇起飞——挑不在乎吵的
#     时段跑（如夜间）：
#       nohup bash scripts/exp/run_t113_to_t116_queue.sh > diagnose_output/queue_nohup.log 2>&1 &
#   - 判定不在本脚本里做：四批跑完后用各自 JSON 做配对判定（判据已预注册在
#     各 run_t11x_*.sh 头部注释）。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
LOG="diagnose_output/queue_driver.log"

say () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_batch () {  # $1=名字 $2=完成标志文件 $3=脚本
  if [ -f "$2" ]; then
    say "[SKIP] $1 已完成（$2 存在）"
    return 0
  fi
  say "[RUN ] $1 开始"
  bash "$3" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    say "[FAIL] $1 exit=$rc —— 队列中止（后续批依赖判定基线，不盲跑）"
    exit $rc
  fi
  say "[OK  ] $1 完成"
}

run_batch "T113 fp16基线"   "diagnose_output/T113_fp16_s37.json"       scripts/exp/run_t113_fp16_baseline.sh
run_batch "T114 因子瘦身"   "diagnose_output/T114_slim_s37.json"       scripts/exp/run_t114_slim.sh
run_batch "T116 门控终检"   "diagnose_output/T116_macro_gate_s37.json" scripts/exp/run_t116_scalar_macro_gate.sh

# T115 前置：idxrel 缓存（CPU，~40-60min；已构建则脚本内部自动全 SKIP）
if [ ! -f "database/system_data/factors_cache_2026-08-18-idxrel/factor_cache_manifest.json" ]; then
  say "[RUN ] idxrel 缓存构建"
  "$PY" -u scripts/build_idxrel_cache.py >> "$LOG" 2>&1 || { say "[FAIL] 缓存构建"; exit 3; }
  say "[OK  ] idxrel 缓存构建完成"
fi
# 冒烟：40 只 × CPU 快速过一遍数据通道，确认新列进面板、index_rel 成族
if ! grep -q "T115_SMOKE_OK" "$LOG" 2>/dev/null; then
  say "[RUN ] T115 冒烟（40 只 CPU 验证特征数/分族）"
  "$PY" -u scripts/exp/exp_nam_gate.py \
    --stocks 40 --years 13 --end 2022-09-05 --disable-gate --target returns \
    --skip-baseline --allow-degenerate-downside-risk \
    --cache-dir database/system_data/factors_cache_2026-08-18-idxrel \
    --folds 0.9:1.0 --drop-groups forecast --epochs 1 --min-epochs 1 \
    --device cpu --seed 42 \
    --plot-dir diagnose_output/nam_gate_T115_smoke \
    --output diagnose_output/T115_smoke.json >> "$LOG" 2>&1 \
    || { say "[FAIL] T115 冒烟"; exit 4; }
  if grep -q "index_rel" diagnose_output/T115_smoke.json; then
    say "T115_SMOKE_OK：index_rel 族已进面板"
  else
    say "[FAIL] 冒烟通过但 JSON 里没有 index_rel 族——新列没被吃进去，先查再跑大批"
    exit 5
  fi
fi
run_batch "T115 指数特征"   "diagnose_output/T115_idxrel_s37.json"     scripts/exp/run_t115_idxrel.sh

say "队列全部完成。判定看各 T11x JSON（配对基线=T113）。"
