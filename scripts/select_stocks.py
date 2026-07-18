"""
量化因子选股脚本
基于训练好的机器学习因子模型，从数据库中筛选股票。
流程：
1. 从数据库读取股票列表及基本信息
2. 根据基础条件预筛选（市值、市盈率、股价等）
3. 批量获取行情数据并计算量化因子
4. 使用 ML 模型预测置信度
5. 按置信度降序输出推荐股票
用法：
    python scripts/select_stocks.py                       # 使用默认参数
    python scripts/select_stocks.py --top 30              # 输出前 30 只
    python scripts/select_stocks.py --min-confidence 65   # 最低置信度 65%
    python scripts/select_stocks.py --model models/lightgbm_factor_model.pkl
    python scripts/select_stocks.py --filter            # 开启基础条件筛选 (默认关闭)
    python scripts/select_stocks.py --workers 8            # 8 线程并行
"""
import os
import sys
import argparse
import sqlite3
import time
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import DATABASE_PATH
from config.factor_config import TrainingConfig
from config.strategy_config import MIN_MARKET_CAP, MAX_PE, MIN_PRICE, MAX_PRICE, INCLUDE_ST, SELECTOR_MARKETS
from config.automation_config import AUTO_MODEL_PATH, AUTO_NORM_STATS_PATH
from core.factors.ml_factor_model import MLFactorModel
from core.factors.train_ml_model import MLModelTrainer
warnings.filterwarnings('ignore')
# ============================================================================
# 常量 & 默认配置
# ============================================================================
DEFAULT_MODEL_PATH = AUTO_MODEL_PATH
DEFAULT_MIN_CONFIDENCE = 0
DEFAULT_TOP_N = 20
DEFAULT_LOOKBACK_DAYS = 500        # 获取最近 N 天行情用于因子计算
MIN_DATA_ROWS = 35                 # 最少需要的行情数据条数 (与 ml_factor_strategy.py 一致)
DEFAULT_WORKERS = 15                # 默认并行线程数
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, TrainingConfig.CACHE_DIR)
# ============================================================================
# 辅助函数
# ============================================================================
def find_latest_model(base_dir: str) -> Optional[str]:
    """
    在指定目录下寻找最新的模型。
    如果是目录，寻找该目录下最深层的 pkl 文件。
    """
    if not os.path.exists(base_dir):
        return None
    
    if os.path.isfile(base_dir) and base_dir.endswith('.pkl'):
        return base_dir
        
    # 寻找子目录
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        # 直接在当前目录找 pkl
        pkls = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.pkl')]
        return sorted(pkls)[-1] if pkls else None
    
    # 按修改时间排序子目录
    subdirs.sort(key=os.path.getmtime, reverse=True)
    
    for sd in subdirs:
        res = find_latest_model(sd)
        if res:
            return res
            
    return None
def load_smart_model(model_path: str):
    """
    智能加载模型（与 ml_factor_strategy.py 中的 _load_smart_model 逻辑对齐）：
    1. 如果是目录，优先尝试加载集成模型（双 pkl），否则加载最新 pkl
    2. 如果是 pkl 文件，优先尝试 EnsembleFactorModel，再回退到 MLFactorModel
    """
    from core.factors.ml_factor_model import MLFactorModel, EnsembleFactorModel

    # 情况 1: 目录
    if os.path.isdir(model_path):
        xgb_path = os.path.join(model_path, 'xgboost_factor_model.pkl')
        lgb_path = os.path.join(model_path, 'lightgbm_factor_model.pkl')

        if os.path.exists(xgb_path) and os.path.exists(lgb_path):
            print(f"📦 检测到双模型目录，正在构建集成模型...")
            m1 = MLFactorModel(model_type='xgboost')
            m1.load_model(xgb_path)
            m2 = MLFactorModel(model_type='lightgbm')
            m2.load_model(lgb_path)
            return EnsembleFactorModel(models=[m1, m2], weights=[0.5, 0.5])

        # 否则寻找最新的 pkl
        latest = find_latest_model(model_path)
        if latest:
            return load_smart_model(latest)
        return None

    # 情况 2: pkl 文件 —— 优先尝试 EnsembleFactorModel，再回退到 MLFactorModel
    if not os.path.exists(model_path):
        return None

    try:
        return EnsembleFactorModel.load_model(model_path)
    except Exception:
        pass
    try:
        m = MLFactorModel()
        m.load_model(model_path)
        return m
    except Exception:
        return None
# ============================================================================
# 数据库辅助
# ============================================================================
def get_db_conn(db_path: str):
    """获取带有关联库的连接 (meta + finance)"""
    conn = sqlite3.connect(db_path)
    db_dir = os.path.dirname(db_path)
    meta_db = os.path.join(db_dir, 'stock_meta.db')
    finance_db = os.path.join(db_dir, 'stock_finance.db')
    if os.path.exists(meta_db):
        conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
    if os.path.exists(finance_db):
        conn.execute(f"ATTACH DATABASE '{finance_db}' AS finance")
    return conn
def get_all_stock_codes(db_path: str) -> List[str]:
    """从数据库获取所有有行情数据的股票代码"""
    conn = get_db_conn(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT code FROM daily_data ORDER BY code")
    codes = [row[0] for row in cursor.fetchall()]
    conn.close()
    return codes
def get_stock_info_map(db_path: str) -> Dict[str, Dict]:
    """
    获取股票基本信息:
    - 名称/ST标记: 来自 meta.stock_info_extended
    - PE/PB (动态): 用最新收盘价 / finance_reports 最新期 EPS/BPS 计算
      这样 PE/PB 是 PIT 的，避免使用快照静态数据
    """
    def safe_float(val):
        try:
            if val is None or val == '' or str(val).lower() == 'none':
                return None
            v = float(val)
            return v if np.isfinite(v) else None
        except (ValueError, TypeError):
            return None
    conn = get_db_conn(db_path)
    # 1. 从 meta 读取基础信息 (直接使用标准的 stock_basic 表)
    try:
        meta_df = pd.read_sql_query(
            "SELECT code, code_name AS name FROM meta.stock_basic", conn
        )
    except Exception as e:
        import traceback
        print(f"  [ERROR] 从 meta.stock_basic 加载元数据失败: {e}")
        traceback.print_exc()
        meta_df = pd.DataFrame(columns=['code', 'name'])
    # 2. 从 finance 表读取最新一期财务数据 (按 code + 最大 pub_date)
    try:
        # 使用 p.stat_date 关联并以 p.pub_date 对齐最新公告
        finance_df = pd.read_sql_query(
            """
            SELECT p.code, p.epsTTM AS EPSJB, p.totalShare, b.liabilityToAsset AS ZCFZL
            FROM finance.profit_ability p
            LEFT JOIN finance.balance_ability b ON p.code = b.code AND p.stat_date = b.stat_date
            INNER JOIN (
                SELECT code, MAX(pub_date) AS max_date
                FROM finance.profit_ability
                GROUP BY code
            ) latest ON p.code = latest.code AND p.pub_date = latest.max_date
            """,
            conn
        )
    except Exception as e:
        import traceback
        print(f"  [ERROR] 从 finance 表加载财务数据失败: {e}")
        traceback.print_exc()
        finance_df = pd.DataFrame(columns=['code', 'EPSJB', 'totalShare', 'ZCFZL'])
    # 3. 获取每只股票最新价格及动态快照 (is_st, pbMRQ)
    try:
        price_df = pd.read_sql_query(
            """
            SELECT code, close, pbMRQ, is_st
            FROM daily_data
            WHERE (code, date) IN (
                SELECT code, MAX(date) FROM daily_data GROUP BY code
            )
            """,
            conn
        )
    except Exception as e:
        import traceback
        print(f"  [ERROR] 从 daily_data 加载行情快照失败: {e}")
        traceback.print_exc()
        price_df = pd.DataFrame(columns=['code', 'close', 'pbMRQ', 'is_st'])
    conn.close()
    # 构建映射
    fin_map = {str(r.code): r for r in finance_df.itertuples()}
    prc_map = {str(r.code): r for r in price_df.itertuples()}
    meta_map = {str(r.code): r for r in meta_df.itertuples()}
    
    # 构建最终 info_map (以 price_df 为基准)
    info_map = {}
    for r in price_df.itertuples():
        code = str(r.code)
        fin = fin_map.get(code)
        meta = meta_map.get(code)
        
        close = r.close
        eps = getattr(fin, 'EPSJB', None)
        total_share = getattr(fin, 'totalShare', None)
        
        dynamic_pe = close / eps if close and eps and eps > 0 else None
        # market_cap 以 “亿” 为单位
        mcap = (close * total_share / 1e8) if close and total_share else None
        
        # 优先使用 daily_data 中的 is_st，因为它更及时
        is_st = getattr(r, 'is_st', 0)
        if is_st == 0 and meta:
            is_st = getattr(meta, 'is_st', 0)
        info_map[code] = {
            'name':          getattr(meta, 'name', '-'),
            'market_cap':    mcap,
            'pe_ratio':      dynamic_pe,
            'pb_ratio':      getattr(r, 'pbMRQ', None), # 动态 PB
            'zcfzl':         getattr(fin, 'ZCFZL', None),
            'current_price': close,
            'is_st':         int(is_st or 0),
        }
    return info_map
# ============================================================================
# 增量缓存更新辅助函数
# ============================================================================
def _update_factor_cache_incremental(db_path: str, codes: List[str], cache_dir: str, workers: int = 12, lookback_days: int = 500, target_features: Optional[List[str]] = None):
    """更清晰的增量缓存更新实现"""
    from core.data.market_sentiment_calculator import MarketSentimentCalculator
    
    # 0. 先更新市场情绪因子（全局性指标，只需计算一次）
    print("   [市场情绪] 正在检查并更新全市场情绪指标...")
    sentiment_calc = MarketSentimentCalculator(db_path)
    sentiment_calc.check_and_update()
    
    trainer = MLModelTrainer(db_path=db_path)
    # 使用传入的 cache_dir，若未指定则回退到默认目录
    effective_cache_dir = os.path.abspath(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    trainer.factors_cache_dir = effective_cache_dir
    os.makedirs(effective_cache_dir, exist_ok=True)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    print(f"   [因子同步] 正在为 {len(codes)} 只股票从数据库加载行情 (缓存: {effective_cache_dir}) ...")
    stocks_data = trainer.load_training_data(codes, start_date, end_date)
    
    if not stocks_data:
        print("   ✗ 未获取到有效行情数据，同步跳过")
        return

    # 直接调用统一的批量更新方法
    trainer.batch_update_factor_cache(
        stocks_data=stocks_data,
        n_jobs=workers,
        target_features=target_features,
        verbose=False
    )
# ============================================================================
# 主流程
# ============================================================================
def select_stocks(
    model_path: str = DEFAULT_MODEL_PATH,
    norm_stats_path: Optional[str] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    top_n: int = DEFAULT_TOP_N,
    apply_filter: bool = False,
    workers: int = DEFAULT_WORKERS,
    cache_dir: str = DEFAULT_CACHE_DIR,
    only_cache: bool = False,
    save_csv: bool = True,
    skip_cache_update: bool = True,
    # 动态过滤参数
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    max_zcfzl: Optional[float] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    include_st: Optional[bool] = None,
    markets: Optional[List[str]] = None,
) -> List[Dict]:
    """
    执行完整的选股流程。直接复用 MLFactorBacktestStrategy.select_for_live()，
    保证与回测逻辑完全一致。

    参数优先级：后端场景 - 前端传入的参数生效，未传入或为 None 则不限制该条件

    参数:
        model_path:      训练好的模型文件路径
        min_confidence:  最小置信度阈值（百分制）
        top_n:           输出前 N 只股票
        apply_filter:    是否使用基础条件预筛选（后端必须显式传入）
        workers:         并行线程数（用于缓存更新）
        cache_dir:       因子缓存目录
        only_cache:      保留参数，暂不使用（缓存由策略内部管理）
        save_csv:        是否将结果保存为 CSV
        skip_cache_update: 跳过增量缓存更新
        min_market_cap:  (动态) 最小市值（亿）- None 表示不限制
        max_pe:          (动态) 最大市盈率 - None 表示不限制
        max_zcfzl:       (动态) 最大资产负债率 (%) - None 表示不限制
        min_price:       (动态) 最小价格 - None 表示不限制
        max_price:       (动态) 最大价格 - None 表示不限制
        include_st:      (动态) 是否包含 ST - None 表示不限制
        markets:         (动态) 市场类型列表 - None 表示不限制
    返回:
        按置信度降序的候选股票列表
    """
    from core.backtest.strategies.ml_factor_strategy import MLFactorBacktestStrategy

    t_start = time.time()

    # 路径标准化
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return []

    if norm_stats_path is None:
        auto_model_abs = os.path.abspath(os.path.join(PROJECT_ROOT, AUTO_MODEL_PATH))
        if os.path.abspath(model_path) == auto_model_abs:
            norm_stats_path = AUTO_NORM_STATS_PATH
    if norm_stats_path and not os.path.isabs(norm_stats_path):
        norm_stats_path = os.path.join(PROJECT_ROOT, norm_stats_path)

    print("=" * 80)
    print("📊 量化因子选股系统")
    print("=" * 80)
    print(f"\n🕐 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 最低置信度: {min_confidence:.1f}% | top_n: {top_n}")

    # ------------------------------------------------------------------
    # Step 1: 增量更新因子缓存
    # ------------------------------------------------------------------
    if not skip_cache_update:
        print("\n" + "-" * 60)
        print(f"增量更新因子缓存 (目录: {cache_dir}) ...")
        try:
            all_codes = get_all_stock_codes(DATABASE_PATH)
            _update_factor_cache_incremental(
                db_path=DATABASE_PATH,
                codes=all_codes,
                cache_dir=cache_dir,
                workers=workers,
            )
        except Exception as _e:
            import traceback
            print(f"  [警告] 增量缓存更新失败，将直接使用旧缓存: {_e}")
            traceback.print_exc()
    else:
        print("\n跳过增量缓存更新 (--skip-cache-update)")

    # ------------------------------------------------------------------
    # Step 2: 通过回测策略生成信号（与回测/自动化完全对齐）
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("⚖️  通过回测策略生成信号 ...")

    # 始终组装 criteria 并携带 apply_filter 标志
    # 后端场景：前端传入的参数优先，不传或为 None 则不限制该条件
    criteria = {
        'min_market_cap': min_market_cap,
        'max_pe':         max_pe,
        'max_zcfzl':      max_zcfzl,
        'min_price':      min_price,
        'max_price':      max_price,
        'include_st':     include_st,
        'markets':        markets,
        'apply_filter':   apply_filter,  # 后端场景：必须显式传入，不传则为 False
    }

    strategy = MLFactorBacktestStrategy(
        model_path=model_path,
        min_confidence=min_confidence,
        cache_dir=cache_dir,
        norm_stats_path=norm_stats_path,
    )
    try:
        strategy.initialize()
        raw_signals = strategy.select_for_live(
            db_path=DATABASE_PATH,
            top_n=top_n,
            criteria=criteria,
        )
    finally:
        strategy.cleanup()

    if not raw_signals:
        print("❌ 未获取到任何信号。")
        return []

    # ------------------------------------------------------------------
    # Step 3: 补全前后端所需字段，保持返回格式兼容
    # ------------------------------------------------------------------
    info_map = get_stock_info_map(DATABASE_PATH)
    today_str = datetime.now().strftime('%Y-%m-%d')

    results = []
    for sig in raw_signals:
        code = sig['stock_code']
        confidence = sig['confidence']
        info = info_map.get(code, {})
        results.append({
            'stock_code':    code,
            'confidence':    confidence,
            'prediction':    confidence / 100.0,
            'current_price': sig['current_price'],
            'latest_date':   today_str,
            'stop_loss':     sig.get('stop_loss'),
            'take_profit':   sig.get('take_profit'),
            'name':          info.get('name', '-'),
            'market_cap':    info.get('market_cap'),
            'pe_ratio':      info.get('pe_ratio'),
            'pb_ratio':      info.get('pb_ratio'),
            'signal':        'buy' if confidence >= max(min_confidence, 50.0) else 'hold',
        })

    # ------------------------------------------------------------------
    # Step 4: 结果展示与保存
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"🏆 选股完成! 共 {len(results)} 只 | 耗时 {elapsed:.1f}s")
    print("=" * 80)

    print(f"{'排名':>2} {'代码':<8} {'名称':<6} {'置信度':>6} {'现价':>7} {'止损':>7} {'止盈':>7}")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        sl = f"{r['stop_loss']:.2f}" if r['stop_loss'] else '  N/A '
        tp = f"{r['take_profit']:.2f}" if r['take_profit'] else '  N/A '
        print(f"{i:>2}. {r['stock_code']:<8} {str(r['name'])[:6]:<6} "
              f"{r['confidence']:>6.2f}% {r['current_price']:>7.2f} {sl:>7} {tp:>7}")

    if save_csv:
        output_dir = os.path.join(PROJECT_ROOT, 'backtest_result')
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"selected_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 结果 CSV 已保存: {csv_path}")

    # 显示因子重要性（从策略模型中读取）
    try:
        tmp_model = load_smart_model(model_path)
        if tmp_model:
            top_factors = tmp_model.get_top_factors(n=10)
            if top_factors:
                print(f"\n🔑 决策核心因子 Top-10:")
                for rank, (fname, val) in enumerate(top_factors, 1):
                    print(f"   {rank:>2}. {fname:<30} {val:.4f}")
    except Exception:
        pass

    return results
# ============================================================================
# CLI 入口
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="量化因子选股 - 基于机器学习模型筛选 A 股",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/select_stocks.py
  python scripts/select_stocks.py --top 50 --min-confidence 55
  python scripts/select_stocks.py --model models/lightgbm_factor_model.pkl
  python scripts/select_stocks.py --filter --workers 8
        """,
    )
    parser.add_argument(
        '--model', type=str, default=DEFAULT_MODEL_PATH,
        help=f'模型文件路径 (默认: {DEFAULT_MODEL_PATH})',
    )
    parser.add_argument(
        '--norm-stats', type=str, default=None,
        help='归一化统计量路径；默认从模型目录读取，自动化默认模型使用绑定的训练归档',
    )
    parser.add_argument(
        '--min-confidence', type=float, default=DEFAULT_MIN_CONFIDENCE,
        help=f'最小置信度阈值 (默认: {DEFAULT_MIN_CONFIDENCE}%%)',
    )
    parser.add_argument(
        '--top', type=int, default=DEFAULT_TOP_N,
        help=f'输出前 N 只股票 (默认: {DEFAULT_TOP_N})',
    )
    parser.add_argument(
        '--filter', action='store_true',
        help='应用基础条件预筛选 (市值/PE/股价等)',
    )
    parser.add_argument(
        '--workers', type=int, default=DEFAULT_WORKERS,
        help=f'并行线程数 (默认: {DEFAULT_WORKERS})',
    )
    parser.add_argument(
        '--cache-dir', type=str, default=DEFAULT_CACHE_DIR,
        help=f'因子缓存目录 (默认: {DEFAULT_CACHE_DIR})',
    )
    parser.add_argument(
        '--only-cache', action='store_true', default=False,
        help='强制只从缓存中读取因子 (忽略数据库更新)',
    )
    parser.add_argument(
        '--no-only-cache', action='store_false', dest='only_cache',
        help='如果缓存不存在则重新计算因子',
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='不保存 CSV 文件',
    )
    parser.add_argument(
        '--skip-cache-update', action='store_true', default=True,
        help='跳过增量缓存更新步骤，直接使用已有缓存（速度更快，但因子可能非最新）',
    )
    # 动态过滤参数（与 select_stocks() 函数签名对齐）
    parser.add_argument('--min-market-cap', type=float, default=None,
                        help='最小流通市值（亿元），None 表示不限制')
    parser.add_argument('--max-pe', type=float, default=None,
                        help='最大市盈率，None 表示不限制')
    parser.add_argument('--max-zcfzl', type=float, default=None,
                        help='最大资产负债率（%%），None 表示不限制')
    parser.add_argument('--min-price', type=float, default=None,
                        help='最小股价（元），None 表示不限制')
    parser.add_argument('--max-price', type=float, default=None,
                        help='最大股价（元），None 表示不限制')
    parser.add_argument('--include-st', action='store_true', default=None,
                        help='包含 ST / *ST 股票')
    parser.add_argument('--exclude-st', action='store_false', dest='include_st',
                        help='排除 ST / *ST 股票')
    parser.add_argument('--markets', type=str, default=None,
                        help='市场类型，逗号分隔，如 sh_main,sz_main')
    return parser.parse_args()



def main():
    args = parse_args()
    # 解析 markets 参数（逗号分隔字符串 -> 列表）
    markets = [m.strip() for m in args.markets.split(',')] if args.markets else None
    select_stocks(
        model_path=args.model,
        norm_stats_path=args.norm_stats,
        min_confidence=args.min_confidence,
        top_n=args.top,
        apply_filter=args.filter,
        workers=args.workers,
        cache_dir=args.cache_dir,
        only_cache=args.only_cache,
        save_csv=not args.no_save,
        skip_cache_update=args.skip_cache_update,
        min_market_cap=args.min_market_cap,
        max_pe=args.max_pe,
        max_zcfzl=args.max_zcfzl,
        min_price=args.min_price,
        max_price=args.max_price,
        include_st=args.include_st,
        markets=markets,
    )
if __name__ == '__main__':
    main()
