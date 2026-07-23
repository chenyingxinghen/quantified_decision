"""
并行回测驱动器

将多个回测任务作为独立子进程并发运行 scripts/run_backtest.py，
通过线程池信号量控制并发数，避免同时加载多份因子缓存导致内存爆炸。
每个子回测的结果各自落盘，最后汇总退出码。

用法示例:
  python -u scripts/run_backtests_parallel.py \
      --model models/latest/multi_objective_factor_model.pkl \
      --start 2026-01-23 --end 2026-07-22 \
      --plan "rank_y_ret_5d:5,rank_y_ret_20d:20,rank_y_ret_60d:60,rank_y_mdd_20d:20,rank_y_downvol_20d:20,rank_y_illiq_20d:20" \
      --max-parallel 2 --time-stop-max-return 10.0 \
      --output-base backtest_result/by_objective
"""
import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(spec, args):
    obj, days = spec
    out = os.path.join(args.output_base, f"{obj}_hold{days if days is not None else 'def'}")
    cmd = [
        sys.executable, '-u', os.path.join(PROJECT_ROOT, 'scripts', 'run_backtest.py'),
        '--model', args.model,
        '--start', args.start,
        '--end', args.end,
        '--objective', obj,
        '--time-stop-max-return', str(args.time_stop_max_return),
        '--output', out,
    ]
    if days is not None:
        cmd += ['--hold-days', str(days)]
    print(f"[parallel] START {obj} hold={days} -> {out}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    dt = time.time() - t0
    print(f"[parallel] DONE  {obj} hold={days} exit={rc} ({dt:.1f}s)", flush=True)
    return (obj, days, rc)


def main():
    parser = argparse.ArgumentParser(description='并行回测驱动器')
    parser.add_argument('--model', required=True, help='模型文件路径')
    parser.add_argument('--start', required=True, help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='回测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--plan', required=True,
                        help='回测计划: "obj:hold,obj:hold,..." 例如 "rank_y_ret_5d:5,rank_y_ret_20d:20"')
    parser.add_argument('--max-parallel', type=int, default=2,
                        help='最大并发回测子进程数 (默认2; 按机器内存量力调大)')
    parser.add_argument('--time-stop-max-return', type=float, default=10.0,
                        help='时间止损最大收益率阈值 (设很大即到点必退，实现固定持仓)')
    parser.add_argument('--output-base', default='backtest_result/by_objective',
                        help='各回测结果输出基目录')
    args = parser.parse_args()

    plan = []
    for item in args.plan.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            obj, days = item.split(':', 1)
            plan.append((obj.strip(), int(days.strip())))
        else:
            plan.append((item.strip(), None))
    if not plan:
        print("错误: --plan 为空")
        sys.exit(2)

    max_par = max(1, min(args.max_parallel, len(plan)))
    print(f"=== 并行回测启动: {len(plan)} 个任务, 最大并发={max_par} ===")

    results = []
    with ThreadPoolExecutor(max_workers=max_par) as ex:
        futs = [ex.submit(run_one, spec, args) for spec in plan]
        for f in as_completed(futs):
            results.append(f.result())

    failed = [r for r in results if r[2] != 0]
    print(f"\n并行回测完成: 共 {len(results)} 个, 成功 {len(results) - len(failed)}, 失败 {len(failed)}")
    for obj, days, rc in sorted(results, key=lambda x: str(x[0])):
        print(f"  {obj} hold={days} -> exit={rc}")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
