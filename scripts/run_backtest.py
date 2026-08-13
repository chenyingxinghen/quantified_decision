"""
使用新回测系统运行回测

演示如何使用core.backtest模块进行回测
"""

import hashlib
import os
import re
import sys

import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.backtest import BacktestEngine, DataHandler, PerformanceAnalyzer
from core.data.baostock_main import BaostockDataManager
from core.backtest.strategies import MLFactorBacktestStrategy
from config import DATABASE_PATH,TrainingConfig
from config.strategy_config import (
    ML_FACTOR_MIN_CONFIDENCE,
    ML_FACTOR_MODEL_PATH,
    INITIAL_CAPITAL,
    COMMISSION_RATE,
    BUY_COST_RATE,
    SELL_COST_RATE,
    MAX_POSITIONS,
    SELECTOR_MARKETS, ENABLE_FUNDAMENTAL_FILTER,
    ML_FACTOR_RISK_MIN_PRICE,
    ML_FACTOR_RISK_EXCLUDE_ST,
)
from datetime import datetime, timedelta


def main():
    """主函数"""
    print("=" * 80)
    print("回测系统")
    print("=" * 80)
    
    # 命令行参数：允许固定模型与回测区间，保证多模型对照完全一致。
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument('--model', type=str, default=None, help='模型路径(pkl文件或目录)，覆盖配置默认值')
    _parser.add_argument('--start', type=str, default=None, help='回测开始日期 (YYYY-MM-DD)')
    _parser.add_argument('--end', type=str, default=None, help='回测结束日期 (YYYY-MM-DD)')
    _parser.add_argument(
        '--min-confidence',
        type=float,
        default=ML_FACTOR_MIN_CONFIDENCE,
        help='最低模型置信度（百分制）',
    )
    _parser.add_argument(
        '--risk-min-price',
        type=float,
        default=ML_FACTOR_RISK_MIN_PRICE,
        help='独立风险过滤：最低当日原始收盘价',
    )
    _parser.add_argument(
        '--no-risk-min-price',
        action='store_const',
        const=None,
        dest='risk_min_price',
        help='关闭独立最低价风险过滤',
    )
    _st_group = _parser.add_mutually_exclusive_group()
    _st_group.add_argument('--exclude-st', action='store_true', dest='exclude_st', help='独立风险过滤：排除当日 ST 股票')
    _st_group.add_argument('--include-st', action='store_false', dest='exclude_st', help='关闭独立 ST 风险过滤')
    _parser.set_defaults(exclude_st=ML_FACTOR_RISK_EXCLUDE_ST)
    _parser.add_argument('--tag', type=str, default=None, help='结果目录附加标签，用于区分实验流水线')
    _parser.add_argument('--max-positions', type=int, default=None,
                        help='覆盖最大持仓数（默认用 sc.MAX_POSITIONS）；设为 20 即回测 Top-20 头部区间策略')
    _parser.add_argument('--max-rel-atr', type=float, default=None,
                        help='风控层：入场前波动率上限 (ATR14/close)，如 0.05；不传则关闭')
    _parser.add_argument('--regime-filter', type=str, default='off', choices=['off', 'trend'],
                        help='风控层：regime 空仓开关。trend=指数在MA20下方且均线下行时停止开仓')
    _parser.add_argument('--risk-penalty-lambda', type=float, default=0.0,
                        help='打分层风险惩罚强度：score_pct - lambda*risk_pct；0 表示关闭')
    _parser.add_argument('--risk-penalty-feature', type=str, default='max_drawdown_20',
                        help='风险惩罚使用的 PIT 因子列，默认 max_drawdown_20')
    _parser.add_argument('--risk-penalty-direction', type=str, default='low', choices=['high', 'low'],
                        help='风险方向：high=值越高越危险；low=值越低越危险（负回撤列）')
    _parser.add_argument('--ensemble-model', action='append', default=[],
                        help='追加一个等权横截面分位集成子模型；可重复传入')
    _parser.add_argument('--cache-dir', type=str, default=None,
                        help='显式覆盖因子缓存目录；模型有绑定清单时必须版本一致')
    _parser.add_argument('--buy-cost', type=float, default=None,
                        help='买入单边总成本率（佣金+规费+滑点），默认取 sc.BUY_COST_RATE')
    _parser.add_argument('--sell-cost', type=float, default=None,
                        help='卖出单边总成本率（佣金+规费+滑点+印花税），默认取 sc.SELL_COST_RATE')
    _args, _ = _parser.parse_known_args()

    _max_positions = _args.max_positions if _args.max_positions is not None else MAX_POSITIONS

    # 配置参数
    start_date = (
        _args.start
        if _args.start
        else (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)).strftime('%Y-%m-%d')
    )
    end_date = _args.end if _args.end else datetime.now().strftime('%Y-%m-%d')
    initial_capital = INITIAL_CAPITAL
    commission_rate = COMMISSION_RATE
    buy_cost_rate = _args.buy_cost if _args.buy_cost is not None else BUY_COST_RATE
    sell_cost_rate = _args.sell_cost if _args.sell_cost is not None else SELL_COST_RATE

    # 模型路径：优先命令行 --model，其次配置默认值
    model_path = _args.model if _args.model else ML_FACTOR_MODEL_PATH
    if not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, model_path)
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行 train_ml_model.py 训练模型")
        return
    
    # 1. 创建数据处理器
    print("\n初始化数据处理器...")
    data_handler = DataHandler(DATABASE_PATH)
    
    # 2. 创建策略
    print("初始化策略...")
    
    # 缓存目录：新模型可在目录清单中绑定独立版本化缓存；旧模型无清单时
    # 兼容历史 TrainingConfig.CACHE_DIR。显式 --cache-dir 仅用于诊断覆盖。
    from core.factors.cache_manifest import resolve_ensemble_cache, validate_cache_manifest
    _cache_model_paths = [model_path] + list(_args.ensemble_model)
    _cache_model_dirs = [
        path if os.path.isdir(path) else os.path.dirname(path)
        for path in _cache_model_paths
    ]
    bound_cache_dir = resolve_ensemble_cache(_cache_model_dirs, TrainingConfig.CACHE_DIR)
    if _args.cache_dir:
        cache_dir = os.path.abspath(_args.cache_dir)
        # 若模型有绑定清单，覆盖目录必须至少是合法当前版本缓存。
        if os.path.abspath(bound_cache_dir) != os.path.abspath(TrainingConfig.CACHE_DIR):
            validate_cache_manifest(cache_dir)
    else:
        cache_dir = bound_cache_dir
    use_cache = os.path.exists(cache_dir) and any(
        name.endswith('.parquet') for name in os.listdir(cache_dir)
    )
    
    if use_cache:
        print(f"  检测到因子缓存目录: {cache_dir}")
        print(f"  将使用缓存进行回测")
    else:
        print(f"  错误: 未检测到因子缓存！回测需要预计算的因子缓存。")
        print(f"  请先运行 train_ml_model.py --cache-engineered 生成因子缓存")
        return
    
    strategy = MLFactorBacktestStrategy(
        model_path=model_path,
        min_confidence=_args.min_confidence,
        use_cache=use_cache,
        cache_dir=cache_dir,
        risk_min_price=_args.risk_min_price,
        risk_exclude_st=_args.exclude_st,
        max_positions=_max_positions,
        max_rel_atr=_args.max_rel_atr,
        regime_filter=_args.regime_filter,
        risk_penalty_lambda=_args.risk_penalty_lambda,
        risk_penalty_feature=_args.risk_penalty_feature,
        risk_penalty_direction=_args.risk_penalty_direction,
        ensemble_model_paths=_args.ensemble_model,
        # R2：因子面板只加载回测窗口内的日期（多留 1 行给 PIT searchsorted），
        # 内存 16 GB → 约 2.5 GB，数值逐位不变。
        preload_start=start_date,
        preload_end=end_date,
        name="ML因子策略",
    )
    
    # 3. 创建回测引擎
    print("初始化回测引擎...")
    engine = BacktestEngine(
        strategy=strategy,
        data_handler=data_handler,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        max_positions=_max_positions,
        buy_cost_rate=buy_cost_rate,
        sell_cost_rate=sell_cost_rate,
    )
    
    # 提前获取股票代码
    stock_codes = None # 这里可以指定，不指定则从DB获取
    if stock_codes is None:
        bdm = BaostockDataManager()
        df_stocks = bdm.get_stock_list_from_db(SELECTOR_MARKETS if ENABLE_FUNDAMENTAL_FILTER else None)
        stock_codes = df_stocks['code'].tolist()
        bdm.close()

        # 不能使用数据库最新日期的价格/ST 状态预裁剪历史股票池。
        # 市场、价格、ST、基本面条件由策略在每个回测日基于当日快照动态判断。


    # 为了让第1个交易日就有足够的历史数据（例如250日均线需要，这里预留365个自然日），
    # 我们加载比回测开始日期更早的数据
    load_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    data_handler.load_data(load_start_date, end_date, stock_codes)
    
    results = engine.run(
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        verbose=True
    )
    
    # 5. 保存结果
    print("\n保存结果...")
    
    # 自动解析模型元数据以进行归档
    # 新格式: .../models/{model_abbr}_{forward_days}d_{years}y_{stocks}s_{config_str}_{task_abbr}_{timestamp}/
    # 旧格式: .../models/{weight_status}_{data_volume}/
    # 注意：--model 可能是目录（策略会加载其内最新 pkl），此时归档名要用该目录自身的 basename，
    # 不能用 dirname（会取到父目录，导致不同模型的结果互相覆盖 / 命名错乱）。
    if os.path.isdir(model_path):
        model_dir = os.path.abspath(model_path)
        # 与策略加载逻辑保持一致：取目录内最新修改的 *_factor_model.pkl
        _pkls = [os.path.join(model_dir, f) for f in os.listdir(model_dir)
                 if f.endswith('_factor_model.pkl') or f == 'ensemble_factor_model.pkl']
        _latest = sorted(_pkls, key=os.path.getmtime)[-1] if _pkls else None
        model_name = os.path.basename(_latest) if _latest else ''
        archive_tag = os.path.basename(model_dir.rstrip(os.sep))
    else:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        archive_tag = os.path.basename(model_dir)
    
    # 解析归档目录名
    archive_tag = os.path.basename(model_dir)
    if archive_tag == 'models': # 如果直接放在 models 下
        archive_tag = 'default'
    elif archive_tag == 'latest': # 如果是latest目录，需要找到实际的归档目录
        # 尝试从latest目录中找到实际的模型目录
        models_root = os.path.dirname(model_dir)
        # 查找所有可能的归档目录
        import glob
        archive_dirs = [d for d in os.listdir(models_root) 
                       if os.path.isdir(os.path.join(models_root, d)) and d != 'latest' and d != 'mark']
        if archive_dirs:
            # 按修改时间排序，取最新的
            archive_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(models_root, d)), reverse=True)
            archive_tag = archive_dirs[0]
    
    # 解析模型类别 (如 xgboost)
    model_category = model_name.split('_')[0]
    
    # 构造回测标识 (包含置信度、风险过滤参数和日期)
    variant_tags = []
    if _args.tag:
        safe_tag = re.sub(r'[^A-Za-z0-9_.-]+', '-', _args.tag).strip('.-')
        if not safe_tag:
            raise ValueError('--tag 必须至少包含一个字母、数字、点、下划线或连字符')
        variant_tags.append(safe_tag)
    if _args.risk_min_price is not None:
        price_tag = f"{_args.risk_min_price:g}".replace('.', 'p')
        variant_tags.append(f"minp{price_tag}")
    if _args.exclude_st:
        variant_tags.append("nost")
    # 持仓上限必须进目录名，否则同一模型的 Top-5 / Top-20 结果会互相覆盖
    variant_tags.append(f"mp{_max_positions}")
    # 风控层参数必须进目录名，否则消融各路结果会互相覆盖
    if _args.max_rel_atr is not None:
        variant_tags.append(f"vc{_args.max_rel_atr:g}".replace('.', 'p'))
    if _args.regime_filter != 'off':
        variant_tags.append(f"rg-{_args.regime_filter}")
    if _args.risk_penalty_lambda > 0:
        lambda_tag = f"{_args.risk_penalty_lambda:g}".replace('.', 'p')
        feature_tag = re.sub(r'[^A-Za-z0-9_-]+', '-', _args.risk_penalty_feature).strip('-')
        variant_tags.append(f"rp{lambda_tag}-{feature_tag}-{_args.risk_penalty_direction}")
    if _args.ensemble_model:
        # 集成成员顺序不影响等权分位结果；排序后再指纹，避免同一实验产生多个目录。
        ensemble_members = sorted(os.path.normcase(os.path.abspath(p))
                                  for p in [model_path] + list(_args.ensemble_model))
        ensemble_key = '\0'.join(ensemble_members)
        ensemble_hash = hashlib.sha256(ensemble_key.encode('utf-8')).hexdigest()[:8]
        variant_tags.append(f"ens{len(ensemble_members)}-{ensemble_hash}")
    confidence_tag = f"{_args.min_confidence:g}".replace('.', 'p')
    variant_suffix = f"_{'_'.join(variant_tags)}" if variant_tags else ""
    backtest_tag = f"conf{confidence_tag}{variant_suffix}_{start_date}_to_{end_date}"
    
    # 创建归档路径: backtest_result/{archive_tag}/{model_category}/{backtest_tag}/
    result_dir = os.path.join('backtest_result', archive_tag, model_category, backtest_tag)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    # 保存交易记录
    PerformanceAnalyzer.save_trades_to_csv(
        results['trades'],
        os.path.join(result_dir, 'backtest_trades.csv')
    )

    # 保存指标摘要与资金曲线（供模型对照分析；逐笔朴素复算会失真，必须用内部资金曲线）
    import json as _json
    with open(os.path.join(result_dir, 'backtest_metrics.json'), 'w', encoding='utf-8') as _mf:
        _json.dump(results.get('metrics', {}), _mf, ensure_ascii=False, indent=2, default=str)
    _eq = results.get('equity_curve')
    if _eq:
        try:
            _eq_df = pd.DataFrame(_eq, columns=['date', 'equity'])
            _eq_df.to_csv(os.path.join(result_dir, 'backtest_equity_curve.csv'), index=False)
        except Exception:
            pass

    # 基准对照：全市场等权。区分 alpha 与 beta，否则无法判断收益来自选股还是市场。
    try:
        from core.backtest.benchmark import (
            build_equal_weight_benchmark, build_buy_hold_equal_weight_benchmark,
            compute_relative_metrics, print_relative_summary,
        )
        _bench_all = {
            '全市场等权(日频再平衡)': build_equal_weight_benchmark(start_date, end_date),
            '全市场等权(买入持有)': build_buy_hold_equal_weight_benchmark(start_date, end_date),
        }
        _rel_all = {}
        for _bname, _bench in _bench_all.items():
            _rel = compute_relative_metrics(_eq or [], _bench)
            print_relative_summary(_rel, _bname)
            _rel_all[_bname] = _rel
            if len(_bench) > 0:
                _fn = 'benchmark_daily_return_%s.csv' % ('rebal' if '再平衡' in _bname else 'bh')
                _bench.to_frame().to_csv(os.path.join(result_dir, _fn))
        with open(os.path.join(result_dir, 'backtest_benchmark.json'), 'w', encoding='utf-8') as _bf:
            _json.dump(_rel_all, _bf, ensure_ascii=False, indent=2, default=str)
    except Exception as _be:
        print(f"\n[警告] 基准对照计算失败: {_be}")

    # 绘制资金曲线
    PerformanceAnalyzer.plot_equity_curve(
        results['equity_curve'],
        title=f"Backtest Equity Curve ({strategy.name})",
        save_path=os.path.join(result_dir, 'backtest_equity.png')
    )
    
    # 绘制置信度与收益率的关系
    PerformanceAnalyzer.plot_confidence_performance(
        results['trades'],
        title=f"Confidence vs. Return ({strategy.name})",
        save_path=os.path.join(result_dir, 'backtest_confidence_analysis.png')
    )
    
    print("\n回测完成！")
    print(f"存档目录: {result_dir}")

    # 6. 清理
    data_handler.close()


if __name__ == '__main__':
    main()
