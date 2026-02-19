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
    python scripts/select_stocks.py --filter            # 跳过基础条件筛选
    python scripts/select_stocks.py --workers 8            # 8 线程并行
"""

import os
import sys
import argparse
import sqlite3
import time
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DATABASE_PATH, SELECTION_CRITERIA
from config.factor_config import TrainingConfig, FactorConfig
from config.strategy_config import ML_FACTOR_MODEL_PATH
from core.factors.ml_factor_model import MLFactorModel
from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator

warnings.filterwarnings('ignore')


# ============================================================================
# 常量 & 默认配置
# ============================================================================

DEFAULT_MODEL_PATH = 'models/unweighted_5y_3193s-3future-day_5profit/xgboost_factor_model.pkl'
DEFAULT_MIN_CONFIDENCE = 0
DEFAULT_TOP_N = 10
DEFAULT_LOOKBACK_DAYS = 500        # 获取最近 N 天行情用于因子计算
MIN_DATA_ROWS = 100                # 最少需要的行情数据条数
DEFAULT_WORKERS = 15                # 默认并行线程数
DEFAULT_CACHE_DIR = TrainingConfig.CACHE_DIR


# ============================================================================
# 数据库辅助
# ============================================================================

def get_all_stock_codes(db_path: str) -> List[str]:
    """从数据库获取所有有行情数据的股票代码"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT code FROM daily_data ORDER BY code")
    codes = [row[0] for row in cursor.fetchall()]
    conn.close()
    return codes


def get_stock_info_map(db_path: str) -> Dict[str, Dict]:
    """
    从 stock_info_extended 表获取股票基本信息
    """
    def safe_float(val):
        try:
            if val is None or val == '' or str(val).lower() == 'none':
                return None
            return float(val)
        except (ValueError, TypeError):
            return None

    conn = sqlite3.connect(db_path)
    try:
        # 优先使用 extended 表
        df = pd.read_sql_query("SELECT * FROM stock_info_extended", conn)
    except Exception:
        try:
            df = pd.read_sql_query("SELECT * FROM stock_info", conn)
        except Exception:
            df = pd.DataFrame()
    conn.close()

    info_map = {}
    if not df.empty:
        for _, row in df.iterrows():
            info_map[row['code']] = {
                'name': row.get('name', ''),
                'market_cap': safe_float(row.get('market_cap')),
                'pe_ratio': safe_float(row.get('pe_ratio')),
                'pb_ratio': safe_float(row.get('pb_ratio')),
                'sector': row.get('sector', '-'),
                'industry': row.get('industry', '-'),
            }
    return info_map


def get_latest_price(db_path: str, code: str) -> Optional[float]:
    """获取股票最新收盘价"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT close FROM daily_data WHERE code = ? ORDER BY date DESC LIMIT 1",
        (code,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_stock_data(db_path: str, code: str, days: int = DEFAULT_LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """
    从数据库中获取指定股票最近 N 天的行情数据
    """
    conn = sqlite3.connect(db_path)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    query = '''
        SELECT date, open, high, low, close, volume, amount
        FROM daily_data
        WHERE code = ? AND date >= ? AND date <= ?
        ORDER BY date ASC
    '''
    df = pd.read_sql_query(query, conn, params=(code, start_date, end_date))
    conn.close()

    if df.empty or len(df) < MIN_DATA_ROWS:
        return None
    return df


# ============================================================================
# 基础条件预筛选
# ============================================================================

def pre_filter_stocks(
    all_codes: List[str],
    info_map: Dict[str, Dict],
    db_path: str,
    criteria: Dict,
) -> Tuple[List[str], Dict]:
    """
    根据 SELECTION_CRITERIA 过滤股票：
    - min_market_cap：最小市值（亿）
    - max_pe：最大市盈率
    - min_price / max_price：股价范围
    """
    def try_float(val):
        try:
            if val is None or val == '' or val == 'None':
                return None
            return float(val)
        except (ValueError, TypeError):
            return None

    min_market_cap = try_float(criteria.get('min_market_cap', 0)) or 0
    max_pe = try_float(criteria.get('max_pe', float('inf'))) or float('inf')
    min_price = try_float(criteria.get('min_price', 0)) or 0
    max_price = try_float(criteria.get('max_price', float('inf'))) or float('inf')

    # 批量获取最新价格
    conn = sqlite3.connect(db_path)
    price_query = '''
        SELECT code, close
        FROM daily_data
        WHERE (code, date) IN (
            SELECT code, MAX(date)
            FROM daily_data
            GROUP BY code
        )
    '''
    try:
        price_df = pd.read_sql_query(price_query, conn)
        price_map = dict(zip(price_df['code'], price_df['close']))
    except Exception:
        price_map = {}
    conn.close()

    passed = []
    skipped_reasons = {'market_cap': 0, 'pe': 0, 'price': 0, 'no_info': 0, 'st': 0}

    for code in all_codes:
        # # 排除 ST / *ST
        # info = info_map.get(code, {})
        # name = info.get('name', '')
        # if name and ('ST' in name.upper() or '退' in name):
        #     skipped_reasons['st'] += 1
        #     continue

        # # 市值筛选
        # market_cap = try_float(info.get('market_cap'))
        # if market_cap is not None and min_market_cap > 0:
        #     if market_cap < min_market_cap:
        #         skipped_reasons['market_cap'] += 1
        #         continue

        # # 市盈率筛选
        # pe = try_float(info.get('pe_ratio'))
        # if pe is not None and max_pe < float('inf'):
        #     if pe <= 0 or pe > max_pe:
        #         skipped_reasons['pe'] += 1
        #         continue

        # 股价筛选
        price = try_float(price_map.get(code))
        if price is not None:
            if price < min_price or price > max_price:
                skipped_reasons['price'] += 1
                continue

        passed.append(code)

    return passed, skipped_reasons


# ============================================================================
# 模型打分
# ============================================================================

def score_single_stock(
    code: str,
    db_path: str,
    factor_calculator: ComprehensiveFactorCalculator,
    model: MLFactorModel,
    cache_dir: Optional[str] = None,
    only_cache: bool = False
) -> Optional[Dict]:
    """
    对单只股票计算因子并用模型打分。

    返回:
        包含 stock_code, signal, confidence, prediction, current_price 等的字典；
        不符合条件或出错时返回 None。
    """
    try:
        # 1. 获取行情数据
        data = get_stock_data(db_path, code)
        if data is None:
            return None

        factors = None
        
        # 2. 尝试从缓存加载因子
        if cache_dir:
            cache_file = os.path.join(cache_dir, f'{code}_factors.parquet')
            if os.path.exists(cache_file):
                try:
                    cached_factors = pd.read_parquet(cache_file)
                    factors = cached_factors
                except:
                    pass


        # 4. 取最新一行因子
        latest = factors.tail(1)
        if latest.isna().all(axis=1).iloc[0]:
            return None

        # 5. 模型预测
        signal_result = model.predict_signal(latest)

        # 6. 收集结果
        current_price = data['close'].iloc[-1]
        latest_date = data['date'].iloc[-1]

        return {
            'stock_code': code,
            'signal': signal_result['signal'],
            'confidence': signal_result['confidence'],
            'prediction': signal_result['prediction'],
            'current_price': current_price,
            'latest_date': latest_date,
        }

    except Exception:
        return None


# ============================================================================
# 主流程
# ============================================================================

def select_stocks(
    model_path: str = DEFAULT_MODEL_PATH,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    top_n: int = DEFAULT_TOP_N,
    apply_filter: bool = False,
    workers: int = DEFAULT_WORKERS,
    cache_dir: str = DEFAULT_CACHE_DIR,
    only_cache: bool = True,
    save_csv: bool = True,
) -> List[Dict]:
    """
    执行完整的选股流程。

    参数:
        model_path:      训练好的模型文件路径
        min_confidence:   最小置信度阈值（百分制）
        top_n:            输出前 N 只股票
        apply_filter:     是否使用 SELECTION_CRITERIA 预筛选
        workers:          并行线程数
        cache_dir:        因子缓存目录
        only_cache:       是否只从缓存加载因子
        save_csv:         是否将结果保存为 CSV

    返回:
        按置信度降序的候选股票列表
    """
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 0: 解析模型路径 & 缓存目录
    # ------------------------------------------------------------------
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先运行训练脚本生成模型。")
        return []
    
    if cache_dir and not os.path.isabs(cache_dir):
        cache_dir = os.path.join(PROJECT_ROOT, cache_dir)

    # ------------------------------------------------------------------
    # Step 1: 加载模型
    # ------------------------------------------------------------------
    print("=" * 80)
    print("📊 量化因子选股系统")
    print("=" * 80)
    print(f"\n🕐 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📈 模型路径: {model_path}")
    print(f"📋 最低置信度: {min_confidence:.1f}%")
    print(f"🔝 输出前: {top_n} 只")
    print(f"🔧 并行线程: {workers}")
    if cache_dir:
        print(f"📂 因子缓存: {cache_dir}")

    model = MLFactorModel()
    model.load_model(model_path)
    print(f"✅ 模型加载成功 (类型={model.model_type}, 任务={model.task}, "
          f"特征数={len(model.feature_names)}, 阈值={model.optimal_threshold:.2f})")

    # ------------------------------------------------------------------
    # Step 2: 获取股票列表 & 基本信息
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📂 加载股票列表 & 基本信息 ...")
    all_codes = get_all_stock_codes(DATABASE_PATH)
    info_map = get_stock_info_map(DATABASE_PATH)
    print(f"   数据库中共 {len(all_codes)} 只股票，stock_info 记录 {len(info_map)} 条")

    # ------------------------------------------------------------------
    # Step 3: 基础条件预筛选
    # ------------------------------------------------------------------
    if apply_filter:
        print("\n📋 应用基础筛选条件:")
        for k, v in SELECTION_CRITERIA.items():
            print(f"   {k} = {v}")

        candidate_codes, skip_reasons = pre_filter_stocks(
            all_codes, info_map, DATABASE_PATH, SELECTION_CRITERIA
        )
        print(f"\n   通过筛选: {len(candidate_codes)} 只")
        print(f"   排除 ST/退市: {skip_reasons['st']}")
        print(f"   排除 市值不足: {skip_reasons['market_cap']}")
        print(f"   排除 PE 不合规: {skip_reasons['pe']}")
        print(f"   排除 股价不在范围: {skip_reasons['price']}")
    else:
        candidate_codes = all_codes
        print(f"\n⚠️  跳过基础筛选，将扫描全部 {len(candidate_codes)} 只股票")

    if not candidate_codes:
        print("❌ 无候选股票，请检查筛选条件或数据库数据。")
        return []

    # ------------------------------------------------------------------
    # Step 4: 多线程计算因子 & 模型打分
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"🔄 开始计算因子并打分（共 {len(candidate_codes)} 只）...")

    # 每个线程需要自己的 ComprehensiveFactorCalculator 实例（内含 DB 连接）
    results: List[Dict] = []
    done_count = 0
    total = len(candidate_codes)

    def _worker(code: str) -> Optional[Dict]:
        # 每个线程创建独立的 calculator 避免 sqlite 线程安全问题
        calc = ComprehensiveFactorCalculator(db_path=DATABASE_PATH)
        return score_single_stock(code, DATABASE_PATH, calc, model, cache_dir=cache_dir, only_cache=only_cache)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_worker, code): code for code in candidate_codes}

        for future in as_completed(future_map):
            done_count += 1
            if done_count % 100 == 0 or done_count == total:
                elapsed = time.time() - t_start
                speed = done_count / elapsed if elapsed > 0 else 0
                print(f"   进度: {done_count}/{total}  "
                      f"({done_count / total * 100:.1f}%)  "
                      f"已选出: {len(results)}  "
                      f"速度: {speed:.1f} 只/秒")

            result = future.result()
            if result and result['confidence'] >= min_confidence:
                results.append(result)

    # ------------------------------------------------------------------
    # Step 5: 排序 & 合并基本信息
    # ------------------------------------------------------------------
    results.sort(key=lambda x: x['confidence'], reverse=True)

    # 截断到 top_n
    results = results[:top_n]

    # 合并 stock_info 信息
    for r in results:
        info = info_map.get(r['stock_code'], {})
        r['name'] = info.get('name', '-')
        r['market_cap'] = info.get('market_cap', None)
        r['pe_ratio'] = info.get('pe_ratio', None)
        r['pb_ratio'] = info.get('pb_ratio', None)

    # ------------------------------------------------------------------
    # Step 6: 展示结果
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"🏆 选股结果 (共 {len(results)} 只, 耗时 {elapsed_total:.1f}s)")
    print("=" * 80)

    if not results:
        print("   本次未筛选到符合条件的股票。")
        return results

    # 表头
    header_fmt = (
        "{:>4}  {:<8}  {:<6}  {:<10}  {:<10}  {:>7}  {:>8}  {:>8}  {:>8}  {:>8}  {:<10}"
    )
    row_fmt = (
        "{:>4}  {:<8}  {:<6}  {:<10}  {:<10}  {:>7.1f}%  {:>8.2f}  {:>8}  {:>8}  {:>8}  {:<10}"
    )
    print(header_fmt.format(
        "排名", "代码", "名称", "板块", "行业", "置信度", "现价",
        "市值(亿)", "PE", "PB", "最新日期"
    ))
    print("-" * 110)

    for i, r in enumerate(results, 1):
        mc_str = f"{r['market_cap']:.0f}" if r['market_cap'] else '-'
        pe_str = f"{r['pe_ratio']:.1f}" if r['pe_ratio'] else '-'
        pb_str = f"{r['pb_ratio']:.2f}" if r['pb_ratio'] else '-'
        sector = r.get('sector', '-')[:10]
        industry = r.get('industry', '-')[:10]
        
        print(row_fmt.format(
            i,
            r['stock_code'],
            r['name'][:6],
            sector,
            industry,
            r['confidence'],
            r['current_price'],
            mc_str,
            pe_str,
            pb_str,
            str(r['latest_date'])[:10],
        ))

    # ------------------------------------------------------------------
    # Step 7: 保存为 CSV
    # ------------------------------------------------------------------
    if save_csv and results:
        output_dir = os.path.join(PROJECT_ROOT, 'backtest_result')
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(
            output_dir,
            f"selected_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        df_out = pd.DataFrame(results)
        col_order = [
            'stock_code', 'name', 'confidence', 'prediction',
            'current_price', 'market_cap', 'pe_ratio', 'pb_ratio',
            'signal', 'latest_date',
        ]
        df_out = df_out[[c for c in col_order if c in df_out.columns]]
        df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 结果已保存: {csv_path}")

    # ------------------------------------------------------------------
    # Step 8: 输出 Top 因子重要性
    # ------------------------------------------------------------------
    top_factors = model.get_top_factors(n=15)
    if top_factors:
        print("\n" + "-" * 60)
        print("🔑 模型 Top-15 重要因子:")
        for rank, (fname, imp) in enumerate(top_factors, 1):
            bar = '█' * int(imp / top_factors[0][1] * 30)
            print(f"   {rank:>2}. {fname:<35} {imp:.4f}  {bar}")

    print("\n" + "=" * 80)
    print(f"✅ 选股完成，耗时 {elapsed_total:.1f} 秒")
    print("=" * 80)

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
        '--min-confidence', type=float, default=DEFAULT_MIN_CONFIDENCE,
        help=f'最小置信度阈值 (默认: {DEFAULT_MIN_CONFIDENCE}%%)',
    )
    parser.add_argument(
        '--top', type=int, default=DEFAULT_TOP_N,
        help=f'输出前 N 只股票 (默认: {DEFAULT_TOP_N})',
    )
    parser.add_argument(
        '--filter', action='store_true',
        help='跳过基础条件预筛选 (市值/PE/股价等)',
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
        '--only-cache', action='store_true', default=True,
        help='只从缓存中读取因子 (默认: True)',
    )
    parser.add_argument(
        '--no-only-cache', action='store_false', dest='only_cache',
        help='如果缓存不存在则重新计算因子',
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='不保存 CSV 文件',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    select_stocks(
        model_path=args.model,
        min_confidence=args.min_confidence,
        top_n=args.top,
        apply_filter=args.filter,
        workers=args.workers,
        cache_dir=args.cache_dir,
        only_cache=args.only_cache,
        save_csv=not args.no_save,
    )


if __name__ == '__main__':
    main()
