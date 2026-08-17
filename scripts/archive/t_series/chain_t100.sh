#!/usr/bin/env bash
# 等 T099 的 8 个回测全部打完 .done 标记再跑 T100 —— 两者都吃 CPU，并行会互相拖慢。
# 用标记文件而不是 pgrep：Git-Bash 下 pgrep 匹配不到 nohup 起的脚本（实测秒退）。
cd "$(dirname "$0")/../../.." || exit 1
for i in $(seq 1 240); do
  n=$(ls diagnose_output/.bt099_*.done 2>/dev/null | wc -l)
  [ "$n" -ge 8 ] && break
  sleep 60
done
echo "[CHAIN] T099 完成 $(ls diagnose_output/.bt099_*.done 2>/dev/null | wc -l)/8，$(date '+%H:%M:%S') 启动 T100"
bash scripts/archive/t_series/run_t100_tree_hpo.sh
