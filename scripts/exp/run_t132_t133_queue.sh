#!/usr/bin/env bash
# ③ → ② 串行队列。两者都要独占 GPU，必须排队而不是并发。
#   T132 全 13 年窗生产候选（主机驻留，拿掉显存天花板）
#   T133 头部集中损失 y_scale∈{4,8}（T130 的直接后续）
# 顺序理由：T132 产出的是**生产载体候选**，优先级高于 T133 的探索性新轴；
#   且 T133 要与 T115 同窗配对，与 T132 无依赖，放后面不损失信息。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
LOG_DIR="diagnose_output"

echo "########## 队列开始 $(date '+%F %H:%M:%S') ##########"
for job in run_t132_prod13y_host run_t133_head_loss; do
  echo "===== $job 开始 $(date '+%F %H:%M:%S') ====="
  bash "scripts/exp/${job}.sh" > "${LOG_DIR}/${job}.log" 2>&1
  rc=$?
  echo "===== $job 结束 exit=$rc $(date '+%F %H:%M:%S') ====="
  if [ $rc -ne 0 ]; then
    echo "[QUEUE] $job 失败（exit=$rc），后续作业继续跑 —— 两者互相独立。"
    tail -20 "${LOG_DIR}/${job}.log"
  fi
done
echo "########## 队列结束 $(date '+%F %H:%M:%S') ##########"
