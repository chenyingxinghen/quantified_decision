#!/usr/bin/env bash
# 哨兵：等 T108 的 3 个种子 × 2 窗口共 6 个回测打完标记，或驱动脚本消失。
cd "$(dirname "$0")/../../.." || exit 1
for i in $(seq 1 720); do   # 最长等 12 小时
  n=$(ls diagnose_output/.bt108_*.done 2>/dev/null | wc -l)
  if [ "$n" -ge 6 ]; then echo "T108 全部完成：$n/6 回测"; exit 0; fi
  if grep -q "^\[DONE\]" diagnose_output/T108_nohup.log 2>/dev/null; then
    echo "T108 驱动结束，完成 $n/6 回测（可能有失败臂）"; exit 0
  fi
  sleep 60
done
echo "T108 等待超时（12 小时），当前 $(ls diagnose_output/.bt108_*.done 2>/dev/null | wc -l)/6"
