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
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DATABASE_PATH
from config.factor_config import TrainingConfig
from config.strategy_config import MIN_MARKET_CAP, MAX_PE, MIN_PRICE, MAX_PRICE, INCLUDE_ST, SELECTOR_MARKETS
from core.factors.ml_factor_model import MLFactorModel
from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
from core.factors.train_ml_model import MLModelTrainer

warnings.filterwarnings('ignore')


# ============================================================================
# 常量 & 默认配置
# ============================================================================

DEFAULT_MODEL_PATH = 'models/mark' # 默认搜寻 mark 目录
DEFAULT_MIN_CONFIDENCE = 0
DEFAULT_TOP_N = 20
DEFAULT_LOOKBACK_DAYS = 500        # 获取最近 N 天行情用于因子计算
MIN_DATA_ROWS = 35                 # 最少需要的行情数据条数 (与 ml_factor_strategy.py 一致)
DEFAULT_WORKERS = 15                # 默认并行线程数
DEFAULT_CACHE_DIR = TrainingConfig.CACHE_DIR


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
    智能加载模型：
    1. 如果是一个 pkl 文件，根据内容加载为 MLFactorModel 或 EnsembleFactorModel
    """
    from core.factors.ml_factor_model import MLFactorModel
    
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

    # 情况 2: pkl 文件
    if not os.path.exists(model_path):
        return None
        
    m = MLFactorModel()
    m.load_model(model_path)
    return m

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

    # 1. 从 meta 读取基础信息
    try:
        meta_df = pd.read_sql_query(
            "SELECT code, name, is_st FROM meta.stock_info_extended", conn
        )
    except Exception:
        meta_df = pd.DataFrame(columns=['code', 'name', 'is_st'])

    # 2. 从 finance_reports 读取最新一期 EPS/BPS (按 code + 最大 REPORT_DATE)
    try:
        finance_df = pd.read_sql_query(
            """
            SELECT f.code, f.EPSJB, f.BPS, f.ZCFZL
            FROM finance.finance_reports f
            INNER JOIN (
                SELECT code, MAX(REPORT_DATE) AS max_date
                FROM finance.finance_reports
                GROUP BY code
            ) latest ON f.code = latest.code AND f.REPORT_DATE = latest.max_date
            """,
            conn
        )
    except Exception:
        finance_df = pd.DataFrame(columns=['code', 'EPSJB', 'BPS', 'ZCFZL'])

    # 3. 获取每只股票最新价格
    try:
        price_df = pd.read_sql_query(
            """
            SELECT code, close
            FROM daily_data
            WHERE (code, date) IN (
                SELECT code, MAX(date) FROM daily_data GROUP BY code
            )
            """,
            conn
        )
    except Exception:
        price_df = pd.DataFrame(columns=['code', 'close'])

    conn.close()

    # 构建映射
    finance_map = {}
    if not finance_df.empty:
        for _, row in finance_df.iterrows():
            finance_map[str(row.get('code', ''))] = {
                'eps': safe_float(row.get('EPSJB')),
                'bps': safe_float(row.get('BPS')),
                'zcfzl': safe_float(row.get('ZCFZL')),
            }

    price_map = {}
    if not price_df.empty:
        for _, row in price_df.iterrows():
            price_map[str(row.get('code', ''))] = safe_float(row.get('close'))

    info_map = {}
    if not meta_df.empty:
        for _, row in meta_df.iterrows():
            code = str(row.get('code', ''))
            fin = finance_map.get(code, {})
            close = price_map.get(code)

            # 动态 PE = 价格 / EPS; 动态 PB = 价格 / BPS
            eps = fin.get('eps')
            bps = fin.get('bps')
            dynamic_pe = None
            dynamic_pb = None
            if close and close > 0:
                if eps and eps > 0:
                    dynamic_pe = close / eps
                if bps and bps > 0:
                    dynamic_pb = close / bps

            info_map[code] = {
                'name':          row.get('name', ''),
                'market_cap':    None,  # market_cap 不在 meta 表中，留 None
                'pe_ratio':      dynamic_pe,
                'pb_ratio':      dynamic_pb,
                'zcfzl':         fin.get('zcfzl'),
                'current_price': close,  # 最新收盘价，供价格筛选使用
                'sector':        '-',
                'industry':      '-',
                'is_st':         int(row.get('is_st', 0) or 0),
            }
    return info_map


def get_latest_price(db_path: str, code: str) -> Optional[float]:
    """获取股票最新收盘价"""
    conn = get_db_conn(db_path)
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
    conn = get_db_conn(db_path)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    query = '''
        SELECT date, open, high, low, close, volume, amount, turnover_rate
        FROM daily_data
        WHERE code = ? AND date >= ? AND date <= ?
        ORDER BY date ASC
    '''
    df = pd.read_sql_query(query, conn, params=(code, start_date, end_date))
    conn.close()

    if df.empty or len(df) < MIN_DATA_ROWS:
        return None
    return df

def pre_filter_stocks(
    all_codes: List[str],
    info_map: Dict[str, Dict],
    db_path: str,
    criteria: Optional[Dict] = None,
) -> Tuple[List[str], Dict]:
    """
    根据 Config Center 过滤股票：
    - min_market_cap：最小市值（亿）
    - max_pe：最大市盈率
    - min_price / max_price：股价范围
    - include_st: 是否包含 ST 股
    - markets: 市场类型过滤 (e.g., 'sh', 'sz_main', 'sz_gem', 'bj')

    criteria 为动态筛选准则（覆盖全局常量），如果不传则使用 strategy_config 中的默认值。
    """
    if criteria is None:
        criteria = {}

    def try_float(val):
        try:
            if val is None or val == '' or val == 'None':
                return None
            return float(val)
        except (ValueError, TypeError):
            return None

    # 以全局常量为基础，动态 criteria 优先覆盖
    min_market_cap = try_float(criteria.get('min_market_cap', MIN_MARKET_CAP)) or 0
    max_pe         = try_float(criteria.get('max_pe', MAX_PE)) or float('inf')
    min_price      = try_float(criteria.get('min_price', MIN_PRICE)) or 0
    max_price      = try_float(criteria.get('max_price', MAX_PRICE)) or float('inf')
    include_st     = bool(criteria.get('include_st', INCLUDE_ST))
    markets_filter = criteria.get('markets', SELECTOR_MARKETS) # Renamed to avoid conflict with 'markets' in pref_map

    # 市场前缀映射
    pref_map = {
        'sh': ('60'),
        'sz_main': ('00'),
        'sz_gem': ('30'),
        'bj': ('8', '4', '9') # Beijing Stock Exchange codes start with 8, 4, 9
    }
    allowed_prefixes = None
    if markets_filter:
        allowed_prefixes = []
        for m in markets_filter:
            p = pref_map.get(m)
            if isinstance(p, tuple):
                allowed_prefixes.extend(p)
            elif p:
                allowed_prefixes.append(p)
        allowed_prefixes = tuple(allowed_prefixes) # Convert to tuple for efficient startswith check

    passed = []
    skipped_reasons = {
        'market_cap': 0, 'pe': 0, 'price': 0, 'no_info': 0,
        'st': 0, 'market': 0, 'zcfzl': 0
    }

    for code in all_codes:
        info = info_map.get(code)
        if not info:
            skipped_reasons['no_info'] += 1
            continue

        name = str(info.get('name', '') or '')

        # 1. 市场类型筛选
        if allowed_prefixes and not code.startswith(allowed_prefixes):
            skipped_reasons['market'] += 1
            continue

        # 2. 排除 ST / *ST / 退市 (除非 include_st=True)
        if not include_st:
            is_st_flag = info.get('is_st', 0)
            if is_st_flag == 1 or (name and '退' in name):
                skipped_reasons['st'] += 1
                continue

        # 3. 市值筛选 (仅在 min_market_cap > 0 且有效时生效; 当前 market_cap 为 None 则跳过)
        if min_market_cap > 0:
            market_cap = try_float(info.get('market_cap'))
            if market_cap is not None and market_cap < min_market_cap:
                skipped_reasons['market_cap'] += 1
                continue

        # 4. 动态 PE 筛选 (价格/EPS, 由 get_stock_info_map 计算)
        pe = try_float(info.get('pe_ratio'))
        if pe is not None:
            if pe <= 0 or pe > max_pe:
                skipped_reasons['pe'] += 1
                continue

        # 5. 资产负债率筛选（金融行业慎用）
        max_zcfzl = try_float(criteria.get('max_zcfzl', None))
        if max_zcfzl is not None:
            zcfzl = try_float(info.get('zcfzl'))
            if zcfzl is not None and zcfzl > max_zcfzl:
                skipped_reasons['zcfzl'] += 1
                continue

        # 6. 股价筛选 (使用 info_map 中的 current_price)
        price = try_float(info.get('current_price'))
        if price is not None:
            if price < min_price or price > max_price:
                skipped_reasons['price'] += 1
                continue

        passed.append(code)

    return passed, skipped_reasons


def get_factors_for_single_stock(
    code: str,
    db_path: str,
    factor_calculator: ComprehensiveFactorCalculator,
    cache_dir: Optional[str] = None,
    only_cache: bool = False
) -> Optional[Dict]:
    """
    对单只股票获取行情数据并计算/提取因子。
    不进行模型打分，仅返回最新一行因子数据。
    """
    try:
        # 1. 获取行情数据
        data = get_stock_data(db_path, code)
        if data is None:
            return None

        factors = None
        cache_file = None
        last_db_date = str(data['date'].iloc[-1])
        
        # 2. 尝试从缓存加载因子并验证时效性
        if cache_dir:
            cache_file = os.path.join(cache_dir, f'{code}_factors.parquet')
            if os.path.exists(cache_file):
                try:
                    df_cache = pd.read_parquet(cache_file)
                    if not df_cache.empty and 'date' in df_cache.columns:
                        last_cache_date = str(df_cache['date'].iloc[-1])
                        # 如果缓存的最新日期与数据库一致，则认为缓存有效
                        if last_cache_date >= last_db_date:
                            factors = df_cache
                except:
                    pass

        # 3. 如果无有效缓存且允许计算，则重新计算因子并保存
        if factors is None and not only_cache:
            factors = factor_calculator.calculate_all_factors(code, data)
            if factors is not None and not factors.empty:
                # 记录日期以支持下次对比 (与 train_ml_model 逻辑对齐)
                factors = factors.copy()
                factors['date'] = data['date'].values
                
                # 保存到缓存，以便下次提速
                if cache_file:
                    try:
                        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                        factors.to_parquet(cache_file, index=False)
                    except Exception as e:
                        pass

        if factors is None or factors.empty:
            return None

        # 4. 取最新一行因子
        latest = factors.tail(1).copy()
        if latest.isna().all(axis=1).iloc[0]:
            return None

        # 5. 收集元数据
        current_price = data['close'].iloc[-1]
        latest_date = data['date'].iloc[-1]

        return {
            'stock_code': code,
            'factors': latest, # 1-row DataFrame
            'current_price': current_price,
            'latest_date': latest_date,
        }

    except Exception:
        import traceback
        # traceback.print_exc()
        return None


# ============================================================================
# 增量缓存更新辅助函数
# ============================================================================

def _update_factor_cache_incremental(
    codes: List[str],
    cache_dir: str,
    db_path: str,
    workers: int = 8,
    lookback_days: int = 365 * 16,  # 加载足够长的历史，确保技术指标的 lookback 窗口足够
) -> None:
    """
    利用 MLModelTrainer.calculate_and_save_factors 的增量逻辑，
    将各股票的因子缓存补齐到当前数据库最新日期。
    
    流程:
    1. 建立一个临时 MLModelTrainer（仅用于调用其因子计算器）
    2. 对每只股票，加载其完整行情历史
    3. 调用增量缓存方法追加新行
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import sqlite3

    trainer = MLModelTrainer(db_path=db_path)
    end_date   = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    # 检查哪些股票缓存已是最新（快速预检）
    conn = sqlite3.connect(db_path)
    try:
        latest_dates = pd.read_sql_query(
            "SELECT code, MAX(date) AS last_date FROM daily_data WHERE code IN ({}) GROUP BY code".format(
                ','.join(['?' for _ in codes])
            ), conn, params=codes
        )
    except Exception:
        latest_dates = pd.DataFrame(columns=['code', 'last_date'])
    finally:
        conn.close()
    
    db_latest = dict(zip(latest_dates['code'], latest_dates['last_date']))

    # 过滤出真正需要更新的股票（缓存最新日期 < 数据库最新日期）
    need_update = []
    for code in codes:
        db_last = db_latest.get(code)
        if db_last is None:
            continue
        cache_file = os.path.join(cache_dir, f'{code}_factors.parquet')
        if not os.path.exists(cache_file):
            need_update.append(code)
            continue
        try:
            cf = pd.read_parquet(cache_file, columns=['date'])
            cache_last = str(cf['date'].max())[:10] if 'date' in cf.columns else ''
            if cache_last < str(db_last)[:10]:
                need_update.append(code)
        except Exception:
            need_update.append(code)

    if not need_update:
        print(f"   全部 {len(codes)} 只股票缓存已是最新，无需更新")
        return

    print(f"   需要增量更新: {len(need_update)}/{len(codes)} 只股票")

    ok_count  = 0
    err_count = 0
    t0 = time.time()

    def _worker_update(code: str):
        try:
            # 加载完整行情历史（技术指标需要足够长的 lookback）
            conn2 = sqlite3.connect(db_path)
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume, amount, turnover_rate FROM daily_data "
                "WHERE code = ? AND date >= ? AND date <= ? ORDER BY date ASC",
                conn2, params=(code, start_date, end_date)
            )
            conn2.close()
            if df.empty or len(df) < 50:
                return 'skip'
            trainer.calculate_and_save_factors(
                code=code, data=df,
                apply_feature_engineering=True,
                verbose=False,
                include_fundamentals=True,
            )
            return 'ok'
        except Exception as e:
            return f'err:{e}'

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_update, c): c for c in need_update}
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if res == 'ok':
                ok_count += 1
            elif res == 'skip':
                pass
            else:
                err_count += 1
            if done % 200 == 0 or done == len(need_update):
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                print(f"   增量更新进度: {done}/{len(need_update)} "
                      f"(成功 {ok_count}, 失败 {err_count}) | {speed:.1f} 只/s")

    print(f"   增量缓存更新完成: {ok_count} 只成功, {err_count} 只失败")


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
    only_cache: bool = False,
    save_csv: bool = True,
    skip_cache_update: bool = False,   # 新增: 跳过增量缓存更新
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
    执行完整的选股流程。

    参数:
        model_path:      训练好的模型文件路径
        min_confidence:   最小置信度阈值（百分制）
        top_n:            输出前 N 只股票
        apply_filter:     是否使用基础条件预筛选
        workers:          并行线程数
        cache_dir:        因子缓存目录
        only_cache:       是否只从缓存加载因子
        save_csv:         是否将结果保存为 CSV
        min_market_cap:   (动态) 最小市值
        max_pe:           (动态) 最大市盈率
        max_zcfzl:        (动态) 最大资产负债率 (%)
        min_price:        (动态) 最小价格
        max_price:        (动态) 最大价格
        include_st:       (动态) 是否包含 ST

    返回:
        按置信度降序的候选股票列表
    """
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 0: 环境准备 & 模型加载
    # ------------------------------------------------------------------
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return []
        
    model = load_smart_model(model_path)
    if model is None:
        print(f"❌ 无法从 {model_path} 加载模型。")
        return []

    print("=" * 80)
    print("📊 量化因子选股系统 (绝对值打分模式)")
    print("=" * 80)
    print(f"\n🕐 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📈 模型类型: {getattr(model, 'model_type', 'unknown')} | 特征数: {len(getattr(model, 'feature_names', []))}")
    print(f"📋 最低置信度: {min_confidence:.1f}% | 并行线程: {workers}")

    # ------------------------------------------------------------------
    # Step 1: 获取全市场股票池 & 基本信息
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📂 加载基础数据 ...")
    all_codes = get_all_stock_codes(DATABASE_PATH)
    info_map = get_stock_info_map(DATABASE_PATH)
    print(f"   数据库中行情覆盖 {len(all_codes)} 只，基本面覆盖 {len(info_map)} 只")

    # ------------------------------------------------------------------
    # Step 2: 基础条件提前过滤 (针对单只股票独立预测，可大幅减负)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📋 执行条件筛选 (PE/市值/价格/ST/市场) ...")
    
    predict_codes = all_codes
    if apply_filter:
        criteria = {
            'min_market_cap': min_market_cap, 'max_pe': max_pe, 
            'max_zcfzl': max_zcfzl, 'min_price': min_price, 
            'max_price': max_price, 'include_st': include_st,
            'markets': markets
        }
        passed_codes, skipped_stats = pre_filter_stocks(all_codes, info_map, DATABASE_PATH)
        predict_codes = passed_codes
        print(f"   筛选完成: 满足条件 {len(predict_codes)} 只 (已过滤 {sum(skipped_stats.values())} 只)")
        if not predict_codes:
            print("❌ 无符合条件的股票进入下一步。")
            return []
    else:
        print(f"   未开启筛选，全量预测")

    # ------------------------------------------------------------------
    # Step 2.5: 增量更新因子缓存（补齐到最新行情日期）
    # ------------------------------------------------------------------
    if not skip_cache_update:
        print("\n" + "-" * 60)
        print("🔄 增量更新因子缓存 (将缓存补齐到最新行情日期) ...")
        try:
            _update_factor_cache_incremental(
                codes=predict_codes,
                cache_dir=cache_dir,
                db_path=DATABASE_PATH,
                workers=workers,
            )
        except Exception as _e:
            print(f"  [警告] 增量缓存更新失败，将直接使用旧缓存: {_e}")
    else:
        print("\n跳过增量缓存更新 (--skip-cache-update)")

    # ------------------------------------------------------------------
    # Step 3: 并行计算因子 (仅针对通过筛选的股票)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"🔄 加载/计算因子数据 (目标: {len(predict_codes)} 只) ...")

    raw_results: List[Dict] = []
    done_count = 0
    total = len(predict_codes)
    factor_calculator = ComprehensiveFactorCalculator(db_path=DATABASE_PATH)
    
    def _worker(code: str) -> Optional[Dict]:
        return get_factors_for_single_stock(code, DATABASE_PATH, factor_calculator, cache_dir=cache_dir, only_cache=only_cache)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_worker, code): code for code in predict_codes}
        for future in as_completed(future_map):
            done_count += 1
            if done_count % 100 == 0 or done_count == total:
                print(f"   进度: {done_count}/{total} ({done_count/total*100:.1f}%) | "
                      f"有效: {len(raw_results)} | 速度: {done_count/(time.time()-t_start):.1f} 只/秒")
            res = future.result()
            if res: raw_results.append(res)

    if not raw_results:
        print("❌ 未获取到任何有效的因子数据。")
        return []

    # ------------------------------------------------------------------
    # Step 4: 批量模型打分 (解耦后的单兵评分)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"⚖️  正在通过模型打分 (样本: {len(raw_results)}) ...")
    
    # 构建特征矩阵
    all_X = pd.concat([r['factors'] for r in raw_results], axis=0, ignore_index=True)
    if 'date' in all_X.columns:
        all_X = all_X.drop(columns=['date'])
    
    # 输入对齐 & 缺失值填充
    all_X_input = all_X.astype(np.float64).fillna(0)
    
    try:
        probs = model.predict(all_X_input)
    except Exception as e:
        print(f"❌ 批量预测失败: {e}")
        return []

    # ------------------------------------------------------------------
    # Step 5: 结果汇总、置信度二次过滤 & 排序
    # ------------------------------------------------------------------
    results = []
    low_confidence_count = 0
    
    for i, r in enumerate(raw_results):
        prob = float(probs[i])
        confidence = prob * 100
        
        # 1. 置信度过滤
        if confidence < min_confidence:
            low_confidence_count += 1
            continue
            
        # 2. 策略一致性过滤 (Optimal Threshold)
        if hasattr(model, 'task') and model.task != 'ranking':
            if prob < getattr(model, 'optimal_threshold', 0.5):
                low_confidence_count += 1
                continue
        
        results.append({
            'stock_code': r['stock_code'],
            'confidence': confidence,
            'prediction': prob,
            'current_price': r['current_price'],
            'latest_date': r['latest_date'],
        })

    print(f"   打分完成: 置信度低于 {min_confidence}% 已排除 {low_confidence_count} 只")

    # 增加哈希扰动作为平局決胜
    import hashlib
    def get_tie_breaker(code):
        return int(hashlib.md5(str(code).encode()).hexdigest(), 16) % 1000 / 100000.0

    # 排序：置信度降序
    results.sort(key=lambda x: (-(x['confidence'] + get_tie_breaker(x['stock_code']))))
    results = results[:top_n]

    # 合并股票名称等信息
    for r in results:
        info = info_map.get(r['stock_code'], {})
        r['name'] = info.get('name', '-')
        r['market_cap'] = info.get('market_cap', None)
        r['pe_ratio'] = info.get('pe_ratio', None)
        r['pb_ratio'] = info.get('pb_ratio', None)
        r['signal'] = 'buy' if r['confidence'] > 50 else 'hold'  # 50% 为中平点，超过即为买入信号

    # ------------------------------------------------------------------
    # Step 6: 结果展示与保存
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"🏆 选股完成! 共 {len(results)} 只 | 耗时 {elapsed:.1f}s")
    print("=" * 80)

    if not results:
        return []

    # 表头打印 (简化展示)
    print(f"{'排名':>2} {'代码':<8} {'名称':<6} {'置信度':>6} {'现价':>7} {'最新日期':<10}")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        print(f"{i:>2}. {r['stock_code']:<8} {str(r['name'])[:6]:<6} {r['confidence']:>6.2f}% {r['current_price']:>7.2f} {str(r['latest_date'])[:10]}")

    if save_csv:
        output_dir = os.path.join(PROJECT_ROOT, 'backtest_result')
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"selected_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 结果 CSV 已保存: {csv_path}")

    # 显示因子重要性
    top_factors = model.get_top_factors(n=10)
    if top_factors:
        print(f"\n🔑 决策核心因子 Top-10:")
        for rank, (name, val) in enumerate(top_factors, 1):
            print(f"   {rank:>2}. {name:<30} {val:.4f}")

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
        '--skip-cache-update', action='store_true', default=False,
        help='跳过增量缓存更新步骤，直接使用已有缓存（速度更快，但因子可能非最新）',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    select_stocks(
        model_path='models/mark/automation/lightgbm_factor_model.pkl',
        min_confidence=args.min_confidence,
        top_n=args.top,
        apply_filter=True,
        workers=args.workers,
        cache_dir=args.cache_dir,
        only_cache=args.only_cache,
        save_csv=not args.no_save,
        skip_cache_update=args.skip_cache_update,
    )


if __name__ == '__main__':
    main()
