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
from config.strategy_config import MIN_MARKET_CAP, MAX_PE, MIN_PRICE, MAX_PRICE, INCLUDE_ST
from core.factors.ml_factor_model import MLFactorModel
from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator

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

    # 情况 2: pkl 文件
    if not os.path.exists(model_path):
        return None
        
    m = MLFactorModel()
    m.load_model(model_path)
    return m

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
        # 全部使用 extended 表
        df = pd.read_sql_query("SELECT * FROM stock_info_extended", conn)
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
                'is_st': row.get('is_st', 0),
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
    criteria: Optional[Dict] = None,
) -> Tuple[List[str], Dict]:
    """
    根据 Config Center 过滤股票：
    - min_market_cap：最小市值（亿）
    - max_pe：最大市盈率
    - min_price / max_price：股价范围
    - include_st: 是否包含 ST 股

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

    # 批量获取最新价格（单次 SQL，避免 N+1 查询）
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
        info = info_map.get(code, {})
        name = str(info.get('name', '') or '')

        # 1. 排除 ST / *ST / 退市 (除非 include_st=True)
        if not include_st:
            # 优先根据数据库 is_st 标签，同时也保留“退”字检查作为补充（针对退市股）
            # 注意：不再单纯根据 'ST' 字符串匹配，因为英文名中 contains 'st' 的单词太多 (如 Industries, Mustang)
            is_st_flag = info.get('is_st', 0)
            if is_st_flag == 1 or (name and '退' in name):
                skipped_reasons['st'] += 1
                continue

        # 2. 市值筛选
        market_cap = try_float(info.get('market_cap'))
        if market_cap is not None:
            if market_cap < min_market_cap:
                skipped_reasons['market_cap'] += 1
                continue

        # 3. 市盈率筛选
        pe = try_float(info.get('pe_ratio'))
        if pe is not None:
            if pe <= 0 or pe > max_pe:
                skipped_reasons['pe'] += 1
                continue

        # 4. 股价筛选
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
        
        # 2. 尝试从缓存加载因子
        if cache_dir:
            cache_file = os.path.join(cache_dir, f'{code}_factors.parquet')
            if os.path.exists(cache_file):
                try:
                    factors = pd.read_parquet(cache_file)
                except:
                    pass

        # 3. 如果没缓存且允许计算，则计算因子
        if factors is None and not only_cache:
            factors = factor_calculator.calculate_all_factors(code, data)

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
    # 新增过滤参数，用于从后端动态传入
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    include_st: Optional[bool] = None,
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
        min_price:        (动态) 最小价格
        max_price:        (动态) 最大价格
        include_st:       (动态) 是否包含 ST

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
        print(f"❌ 模型文件或目录不存在: {model_path}")
        return []
    
    if cache_dir and not os.path.isabs(cache_dir):
        cache_dir = os.path.join(PROJECT_ROOT, cache_dir)

    # ------------------------------------------------------------------
    # Step 1: 加载模型 (智能加载)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("📊 量化因子选股系统")
    print("=" * 80)
    print(f"\n🕐 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📈 目标模型路径: {model_path}")
    print(f"📋 最低置信度: {min_confidence:.1f}%")
    print(f"🔝 输出前: {top_n} 只")
    print(f"🔧 并行线程: {workers}")
    if cache_dir:
        print(f"📂 因子缓存: {cache_dir}")

    model = load_smart_model(model_path)
    if model is None:
        print(f"❌ 无法从 {model_path} 加载模型。")
        return []
        
    m_type = "Ensemble" if hasattr(model, 'models') else getattr(model, 'model_type', 'unknown')
    f_count = len(model.feature_names) if hasattr(model, 'feature_names') else "unknown"
    print(f"✅ 模型加载成功 (类型={m_type}, 特征数={f_count})")

    # ------------------------------------------------------------------
    # Step 2: 获取股票列表 & 基本信息
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📂 加载股票列表 & 基本信息 ...")
    all_codes = get_all_stock_codes(DATABASE_PATH)
    info_map = get_stock_info_map(DATABASE_PATH)
    print(f"   数据库中共 {len(all_codes)} 只股票，stock_info_extended 记录 {len(info_map)} 条")

    # ------------------------------------------------------------------
    # Step 3: 基础条件预筛选
    # ------------------------------------------------------------------
    if apply_filter:
        print("\n📋 应用基础筛选条件:")
        # 构造动态筛选准则，仅使用非 None 的参数覆盖全局默认值
        current_criteria: Dict[str, Any] = {}
        if min_market_cap is not None: current_criteria['min_market_cap'] = min_market_cap
        if max_pe is not None:         current_criteria['max_pe'] = max_pe
        if min_price is not None:      current_criteria['min_price'] = min_price
        if max_price is not None:      current_criteria['max_price'] = max_price
        if include_st is not None:     current_criteria['include_st'] = include_st

        # 打印最终生效的准则（已合并默认值）
        effective = {
            'min_market_cap': current_criteria.get('min_market_cap', MIN_MARKET_CAP),
            'max_pe':         current_criteria.get('max_pe', MAX_PE),
            'min_price':      current_criteria.get('min_price', MIN_PRICE),
            'max_price':      current_criteria.get('max_price', MAX_PRICE),
            'include_st':     current_criteria.get('include_st', INCLUDE_ST),
        }
        for k, v in effective.items():
            print(f"   {k} = {v}")

        candidate_codes, skip_reasons = pre_filter_stocks(
            all_codes, info_map, DATABASE_PATH, current_criteria
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
    raw_results: List[Dict] = []
    done_count = 0
    total = len(candidate_codes)

    def _worker(code: str) -> Optional[Dict]:
        # 每个线程创建独立的 calculator 避免 sqlite 线程安全问题
        calc = ComprehensiveFactorCalculator(db_path=DATABASE_PATH)
        return get_factors_for_single_stock(code, DATABASE_PATH, calc, cache_dir=cache_dir, only_cache=only_cache)

    print(f"   正在多线程获取因子数据...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_worker, code): code for code in candidate_codes}

        for future in as_completed(future_map):
            done_count += 1
            if done_count % 100 == 0 or done_count == total:
                elapsed = time.time() - t_start
                speed = done_count / elapsed if elapsed > 0 else 0
                print(f"   进度: {done_count}/{total}  "
                      f"({done_count / total * 100:.1f}%)  "
                      f"已获取: {len(raw_results)}  "
                      f"速度: {speed:.1f} 只/秒")

            res = future.result()
            if res:
                raw_results.append(res)

    if not raw_results:
        print("❌ 未获取到任何有效的因子数据。")
        return []

    # ------------------------------------------------------------------
    # Step 5: 批量归一化 & 模型打分 (核心：横截面 normalization)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"⚖️  执行横截面 Z-Score 归一化并批量打分...")
    
    # 构造批量特征矩阵
    all_X = pd.concat([r['factors'] for r in raw_results], axis=0, ignore_index=True)
    
    # 移除可能存在的非特征列 (如 date)
    if 'date' in all_X.columns:
        all_X = all_X.drop(columns=['date'])
    
    # 转换为 float64 进行计算
    all_X = all_X.astype(np.float64)
    
    # 执行 Z-Score 归一化 (与 train_ml_model.py L545 & ml_factor_strategy.py L180 一致)
    if len(all_X) > 1:
        # 减去均值，除以标准差。如果标准差为0则替换为1以避免除零，最后填充NaN为0
        all_X_norm = (all_X - all_X.mean()) / all_X.std().replace(0, 1.0)
        all_X_norm = all_X_norm.fillna(0)
    else:
        # 如果只有一只样本，直接设为 0（代表均值水平）
        all_X_norm = all_X * 0
        all_X_norm = all_X_norm.fillna(0)

    # 批量预测
    try:
        probs = model.predict(all_X_norm)
    except Exception as e:
        print(f"❌ 批量预测失败: {e}")
        import traceback
        traceback.print_exc()
        return []

    # 构造最终结果并应用过滤
    results = []
    for i, r in enumerate(raw_results):
        prob = probs[i]
        confidence = float(prob * 100)
        
        # 1. 基础置信度过滤
        if confidence < min_confidence:
            continue
            
        # 2. 策略一致性过滤：如果是回归类模型，应用 optimal_threshold
        # (模仿 ml_factor_strategy.py L201)
        if hasattr(model, 'task') and model.task != 'ranking':
            if prob < getattr(model, 'optimal_threshold', 0.5):
                continue
        
        # 3. 构造结果项
        results.append({
            'stock_code': r['stock_code'],
            'confidence': confidence,
            'prediction': float(prob),
            'current_price': r['current_price'],
            'latest_date': r['latest_date'],
        })

    # ------------------------------------------------------------------
    # Step 6: 排序 & 合并基本信息
    # ------------------------------------------------------------------
    # 引入确定性随机扰动作为平局决胜 (与 ml_factor_strategy.py L229 一致)
    import hashlib
    def get_tie_breaker(code):
        return int(hashlib.md5(str(code).encode()).hexdigest(), 16) % 1000 / 100000.0

    # 1. 排序：置信度降序 + 哈希扰动
    results.sort(key=lambda x: (-(x['confidence'] + get_tie_breaker(x['stock_code']))))

    # 2. 截断到 top_n
    results = results[:top_n]

    # 3. 合并 stock_info 信息
    for r in results:
        info = info_map.get(r['stock_code'], {})
        r['name'] = info.get('name', '-')
        r['market_cap'] = info.get('market_cap', None)
        r['pe_ratio'] = info.get('pe_ratio', None)
        r['pb_ratio'] = info.get('pb_ratio', None)
        
        # 4. 优化：信号标签判定
        # 如果是 Ranking 模型，且已经通过了 top_n 筛选，在这里一律标记为 buy (或根据分数是否 > 均值来判定)
        is_ranking = hasattr(model, 'task') and model.task == 'ranking'
        if is_ranking:
            # Ranking 模型下，只要进入了 TopN 且置信度不为 0，即标记为买入信号
            r['signal'] = 'buy' if r['confidence'] > 0 else 'hold'
        else:
            # 回归模型继续使用 optimal_threshold
            r['signal'] = 'buy' if r['prediction'] >= getattr(model, 'optimal_threshold', 0.5) else 'hold'

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
        
        # 确保名称、板块、行业均为字符串，防止 NaN (float) 导致 upper() 或切片出错
        name_str = str(r.get('name', '-') or '-')
        sector = str(r.get('sector', '-') or '-')[:10]
        industry = str(r.get('industry', '-') or '-')[:10]
        
        print(row_fmt.format(
            i,
            r['stock_code'],
            name_str[:6],
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
