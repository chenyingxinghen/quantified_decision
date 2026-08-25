# -*- coding: utf-8 -*-
"""
R4 内存驻留批量回测驱动。

单一进程内：
  1. 加载一次日线面板 (data_handler.load_data)
  2. 第一个模型把因子面板 (13.85GB parquet) 加载进
     MLFactorBacktestStrategy._GLOBAL_FACTOR_CACHE，后续同指纹模型直接复用
     （同一份 numpy 矩阵，引用共享，不重复加载）
  3. 循环对各模型 engine.run()，每次重置 portfolio，写出与 run_backtest.py
     同结构的产物 (backtest_trades.csv / backtest_metrics.json /
     backtest_equity_curve.csv)

相比 run_backtest.py 逐模型独立进程（每次都重载 13.85GB），
8 个模型回测只需 1 次因子加载，提速约 N 倍（N=模型数）。

产物目录命名与 run_backtest.py 完全一致，因此 diag_random_null.py 与
analyze_sweep_a.py 可直接复用。
"""
import os
import sys
import re
import json
import glob
import argparse
import subprocess
import datetime

ROOT = r"G:/ai_proj/quantified_decision"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import DATABASE_PATH, TrainingConfig  # noqa: E402
from core.backtest import BacktestEngine, PerformanceAnalyzer  # noqa: E402
from core.backtest.baostock_data_handler import BaostockDataHandler  # noqa: E402
from core.backtest.strategies import MLFactorBacktestStrategy  # noqa: E402
from core.backtest.portfolio import Portfolio  # noqa: E402
from core.data.baostock_main import BaostockDataManager  # noqa: E402

NULL_SCRIPT = os.path.join(ROOT, "scripts", "exp", "diag_random_null.py")
PY = r"C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"


def _archive_and_category(model_path):
    """复刻 run_backtest.py 的 archive_tag / model_category 推导。"""
    if os.path.isdir(model_path):
        model_dir = os.path.abspath(model_path)
        pkls = [os.path.join(model_dir, f) for f in os.listdir(model_dir)
                if f.endswith('_factor_model.pkl') or f == 'ensemble_factor_model.pkl']
        latest = sorted(pkls, key=os.path.getmtime)[-1] if pkls else None
        model_name = os.path.basename(latest) if latest else ''
        archive_tag = os.path.basename(model_dir.rstrip(os.sep))
    else:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        archive_tag = os.path.basename(model_dir)
    if archive_tag == 'models':
        archive_tag = 'default'
    elif archive_tag == 'latest':
        models_root = os.path.dirname(model_dir)
        dirs = [d for d in os.listdir(models_root)
                if os.path.isdir(os.path.join(models_root, d))
                and d not in ('latest', 'mark')]
        if dirs:
            dirs.sort(key=lambda d: os.path.getmtime(os.path.join(models_root, d)),
                      reverse=True)
            archive_tag = dirs[0]
    model_category = model_name.split('_')[0] or 'model'
    return archive_tag, model_category


def _build_result_dir(archive_tag, model_category, args, start, end):
    variant_tags = []
    if args.tag:
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '-', args.tag).strip('.-')
        if safe:
            variant_tags.append(safe)
    if args.risk_min_price is not None:
        variant_tags.append(f"minp{args.risk_min_price:g}".replace('.', 'p'))
    if args.exclude_st:
        variant_tags.append("nost")
    variant_tags.append(f"mp{args.max_positions}")
    if args.max_rel_atr is not None:
        variant_tags.append(f"vc{args.max_rel_atr:g}".replace('.', 'p'))
    if args.regime_filter != 'off':
        variant_tags.append(f"rg-{args.regime_filter}")
    conf = f"{args.min_confidence:g}".replace('.', 'p')
    suffix = f"_{'_'.join(variant_tags)}" if variant_tags else ""
    backtest_tag = f"conf{conf}{suffix}_{start}_to_{end}"
    return os.path.join('backtest_result', archive_tag, model_category, backtest_tag)


def run_one(model_path, engine, data_handler, stock_codes, args, start, end):
    """加载模型(复用全局因子缓存) + 回测 + 写出 + 随机零假设。返回 metrics 摘要。"""
    strategy = MLFactorBacktestStrategy(
        model_path=model_path,
        min_confidence=args.min_confidence,
        use_cache=True,
        cache_dir=args.cache_dir,
        risk_min_price=args.risk_min_price,
        risk_exclude_st=args.exclude_st,
        max_positions=args.max_positions,
        max_rel_atr=args.max_rel_atr,
        regime_filter=args.regime_filter,
    )
    strategy.initialize(stock_codes=stock_codes)

    engine.strategy = strategy
    engine.portfolio = Portfolio(
        initial_capital=1.0,
        commission_rate=0.0,
        max_positions=args.max_positions,
        buy_cost_rate=args.buy_cost,
        sell_cost_rate=args.sell_cost,
    )
    results = engine.run(start_date=start, end_date=end,
                         stock_codes=stock_codes, verbose=False)

    archive_tag, model_category = _archive_and_category(model_path)
    result_dir = _build_result_dir(archive_tag, model_category, args, start, end)
    os.makedirs(result_dir, exist_ok=True)

    PerformanceAnalyzer.save_trades_to_csv(
        results['trades'], os.path.join(result_dir, 'backtest_trades.csv'))
    with open(os.path.join(result_dir, 'backtest_metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results.get('metrics', {}), f, ensure_ascii=False,
                  indent=2, default=str)
    eq = results.get('equity_curve')
    if eq:
        try:
            pd = __import__('pandas')
            pd.DataFrame(eq, columns=['date', 'equity']).to_csv(
                os.path.join(result_dir, 'backtest_equity_curve.csv'), index=False)
        except Exception:
            pass

    # 随机零假设（独立进程，只读 trades.csv，快且不吃全局内存）
    csvp = os.path.join(result_dir, 'backtest_trades.csv')
    outn = os.path.join(result_dir, 'random_null.json')
    if os.path.exists(csvp):
        try:
            subprocess.run(
                [PY, NULL_SCRIPT, '--trades', csvp, '--n-sims', str(args.n_sims),
                 '--buy-cost', str(args.buy_cost), '--sell-cost', str(args.sell_cost),
                 '--out', outn], check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception:
            pass

    m = results.get('metrics', {})
    print(f"  [{model_category}/{archive_tag}] "
          f"ret={m.get('total_return_pct'):.2f}% dd={m.get('max_drawdown'):.2f}% "
          f"sharpe={m.get('sharpe_ratio'):.3f} trades={m.get('total_trades')} "
          f"-> {result_dir}")
    return archive_tag, model_category, result_dir, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True,
                    help='多个模型 pkl 路径（目录或文件），同进程常驻回测')
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--max-positions', type=int, default=20)
    ap.add_argument('--min-confidence', type=float, default=0.0)
    ap.add_argument('--risk-min-price', type=float, default=1.0)
    ap.add_argument('--exclude-st', dest='exclude_st', action='store_true',
                    default=True)
    ap.add_argument('--include-st', dest='exclude_st', action='store_false')
    ap.add_argument('--buy-cost', type=float, default=0.0008)
    ap.add_argument('--sell-cost', type=float, default=0.0013)
    ap.add_argument('--max-rel-atr', type=float, default=None)
    ap.add_argument('--regime-filter', default='off')
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--n-sims', type=int, default=2000)
    ap.add_argument('--limit-stocks', type=int, default=None,
                    help='调试用：限制股票池大小')
    args = ap.parse_args()

    # 1. 股票池
    bdm = BaostockDataManager()
    df_stocks = bdm.get_stock_list_from_db(None)
    stock_codes = df_stocks['code'].tolist()
    bdm.close()
    if args.limit_stocks:
        stock_codes = stock_codes[:args.limit_stocks]
    print(f"股票池: {len(stock_codes)} 只")

    # 2. 日线面板只加载一次
    load_start = (datetime.datetime.strptime(args.start, '%Y-%m-%d')
                  - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    data_handler = BaostockDataHandler(DATABASE_PATH)
    print(f"加载日线面板 {load_start} ~ {args.end} ...")
    data_handler.load_data(load_start, args.end, stock_codes)
    print("日线面板已加载，开始批量回测（因子面板将按需全局复用）")

    # 3. engine 复用
    engine = BacktestEngine(
        strategy=MLFactorBacktestStrategy(model_path=args.models[0],
                                          use_cache=True, cache_dir=args.cache_dir),
        data_handler=data_handler,
        initial_capital=1.0,
        commission_rate=0.0,
        max_positions=args.max_positions,
        buy_cost_rate=args.buy_cost,
        sell_cost_rate=args.sell_cost,
    )

    t0 = datetime.datetime.now()
    summary = []
    for i, mpath in enumerate(args.models, 1):
        print(f"\n=== 模型 {i}/{len(args.models)}: {mpath} ===")
        summary.append(run_one(mpath, engine, data_handler, stock_codes,
                               args, args.start, args.end))
    dt = (datetime.datetime.now() - t0).total_seconds()
    print(f"\n=== 批量回测完成 {len(args.models)} 个模型，耗时 {dt:.1f}s ===")


if __name__ == "__main__":
    main()
