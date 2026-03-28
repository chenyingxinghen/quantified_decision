"""
基本面因子模块 (v2 - 基于 finance_reports PIT 数据重构)

数据来源: stock_finance.db -> finance_reports 表
核心字段说明:
  roeAvg / dupontROE - 盈利核心 (ROE)
  npMargin / gpMargin / dupontPnitoni - 利润率
  YOYNI / YOYPNI / YOYEquity - 成长因子
  currentRatio / quickRatio / liabilityToAsset - 偿债及杠杆
  dupontAssetTurn / dupontNitogr - 周转与效率
  epsTTM / totalShare / liqaShare - 每股指标
  dupontTaxBurden / dupontIntburden - 税利负担

PIT 原则:
  - 对于某个交易日 T, 只读取公告日期 NOTICE_DATE <= T 的最近一期报告
  - 这彻底消除了“先看财报结果再选股”的前视偏差 (Data Leakage)
  - 训练时按公告日期对齐, 选股/实时预测时以最新已公布报告为准
  - 对于缺失 NOTICE_DATE 的历史数据，采用保守估算（Q1/Q3+25天, Q2+55天, Q3+25天, Q4+20天）
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from functools import lru_cache

from config.config import DATABASE_PATH


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _safe_float(val, default=np.nan) -> float:
    """安全转换为 float, 失败返回 default"""
    if val is None:
        return default
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def _get_finance_conn(db_path: str) -> sqlite3.Connection:
    """获取携带 finance 附加库的连接"""
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    db_dir = os.path.dirname(db_path)
    finance_db = os.path.join(db_dir, 'stock_finance.db')
    if os.path.exists(finance_db):
        conn.execute(f"ATTACH DATABASE '{finance_db}' AS finance")
    return conn


# ---------------------------------------------------------------------------
# 核心类: FinanceReportFetcher
#   - 从 finance_reports 表读取 PIT 财务数据
#   - 支持单日查询和时间序列对齐
# ---------------------------------------------------------------------------

class FinanceReportFetcher:
    """
    Points-In-Time 财务数据读取器。
    
    - 单日模式: 给定一个截止日期, 返回最近一期报表数据 (dict)
    - 时间序列模式: 给定一个日期序列, 逐日对齐最近期报表, 返回 DataFrame
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        # 预加载所有报告到内存以加速批量查询
        self._cache: Dict[str, pd.DataFrame] = {}

    def _get_conn(self) -> sqlite3.Connection:
        return _get_finance_conn(self.db_path)

    def _load_reports_for_code(self, code: str) -> pd.DataFrame:
        """加载某只股票的全部财务报告 (按 stat_date 升序)"""
        if code in self._cache:
            return self._cache[code]

        conn = self._get_conn()
        try:
            # 基于 stat_date 将 4 张财务表连接为一个宽表
            query = """
                SELECT 
                    p.code, p.pub_date, p.stat_date,
                    p.roeAvg, p.npMargin, p.gpMargin, p.netProfit, p.epsTTM, p.MBRevenue, p.totalShare, p.liqaShare,
                    g.YOYEquity, g.YOYAsset, g.YOYNI, g.YOYEPSBasic, g.YOYPNI,
                    b.currentRatio, b.quickRatio, b.cashRatio, b.YOYLiability, b.liabilityToAsset, b.assetToEquity,
                    d.dupontROE, d.dupontAssetStoEquity, d.dupontAssetTurn, d.dupontPnitoni, d.dupontNitogr, 
                    d.dupontTaxBurden, d.dupontIntburden, d.dupontEbittogr
                FROM finance.profit_ability p
                LEFT JOIN finance.growth_ability g ON p.code = g.code AND p.stat_date = g.stat_date
                LEFT JOIN finance.balance_ability b ON p.code = b.code AND p.stat_date = b.stat_date
                LEFT JOIN finance.dupont d ON p.code = d.code AND p.stat_date = d.stat_date
                WHERE p.code = ?
                ORDER BY p.stat_date ASC
            """
            df = pd.read_sql_query(query, conn, params=(code,))
        except Exception as e:
            import traceback
            print(f"  [ERROR] 加载股票 {code} 财务数据失败: {e}")
            traceback.print_exc()
            df = pd.DataFrame()
        finally:
            conn.close()

        if not df.empty:
            df['stat_date'] = pd.to_datetime(df['stat_date'], errors='coerce')
            df['pub_date']  = pd.to_datetime(df['pub_date'], errors='coerce')
            
            # 估算缺失的公告日期 (pub_date 在 Baostock 中通常是准确的)
            def _fill_notice_date(row):
                if pd.notna(row['pub_date']):
                    return row['pub_date']
                rd = row['stat_date']
                if pd.isna(rd): return pd.NaT
                # 保守预估：Q1->4.25, Q2->8.25, Q3->10.25, Q4->4.20
                if rd.month == 3: return rd.replace(month=4, day=25).replace(year=rd.year)
                if rd.month == 6: return rd.replace(month=8, day=25).replace(year=rd.year)
                if rd.month == 9: return rd.replace(month=10, day=25).replace(year=rd.year)
                if rd.month == 12: return rd.replace(month=4, day=20).replace(year=rd.year+1)
                return rd + pd.Timedelta(days=30)
            
            df['announced_date'] = df.apply(_fill_notice_date, axis=1)
            df = df.dropna(subset=['announced_date']).sort_values('announced_date').reset_index(drop=True)

        self._cache[code] = df
        return df

    def get_pit_report(self, code: str, as_of_date: str) -> Optional[Dict]:
        """
        获取 as_of_date 日期及之前最近一期财报 (单列 dict).
        
        参数:
            code: 6位股票代码, 如 '000001'
            as_of_date: 截止日期字符串 'YYYY-MM-DD'
        
        返回:
            dict 或 None
        """
        df = self._load_reports_for_code(code)
        if df.empty:
            return None

        cutoff = pd.to_datetime(as_of_date)
        valid = df[df['announced_date'] <= cutoff]
        if valid.empty:
            return None

        row = valid.iloc[-1]
        return dict(row)

    def get_pit_series(self, code: str, dates: pd.Series) -> pd.DataFrame:
        """
        为一组日期序列批量对齐 PIT 财务数据。
        
        参数:
            code: 6位股票代码
            dates: 与行情 DataFrame 对应的日期列 (datetime-like 或字符串)
        
        返回:
            与 dates 等长的 DataFrame, 列为财务字段, 无法匹配的行填 NaN
        """
        df = self._load_reports_for_code(code)
        if df.empty:
            return pd.DataFrame(index=dates.index)

        dates_dt = pd.to_datetime(dates)
        announced_dates = df['announced_date'].values  # sorted ascending numpy array
        report_rows = df.drop(columns=['announced_date'], errors='ignore')
        
        # 用 searchsorted 快速找到每个交易日对应的“最新已公布”报告
        indices = np.searchsorted(announced_dates, dates_dt.values, side='right') - 1

        aligned_rows = []
        for idx in indices:
            if idx < 0:
                aligned_rows.append({})
            else:
                aligned_rows.append(dict(report_rows.iloc[idx]))

        result = pd.DataFrame(aligned_rows, index=dates.index)
        return result


# ---------------------------------------------------------------------------
# 核心类: FundamentalFactors
#   - 基于 PIT 财务数据, 构建多类别因子
#   - 在行情 DataFrame 上生成 Rolling 基本面特征
# ---------------------------------------------------------------------------

class FundamentalFactors:
    """
    基本面因子计算器 (PIT 版本)
    
    使用方式:
      ff = FundamentalFactors(db_path)
      # 单股完整时间序列
      factors_df = ff.calculate_fundamental_series(code, daily_data)
      # 仅取最新一期（选股/实盘）
      factors_dict = ff.get_latest_fundamental_factors(code)
    """

    # 所有财务字段 (映射至 Baostock 列名)
    NUMERIC_COLS = [
        'roeAvg', 'npMargin', 'gpMargin', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare',
        'YOYEquity', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI',
        'currentRatio', 'quickRatio', 'cashRatio', 'YOYLiability', 'liabilityToAsset', 'assetToEquity',
        'dupontROE', 'dupontAssetStoEquity', 'dupontAssetTurn', 'dupontPnitoni', 'dupontNitogr', 
        'dupontTaxBurden', 'dupontIntburden', 'dupontEbittogr'
    ]

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.fetcher = FinanceReportFetcher(db_path)

    # ------------------------------------------------------------------
    # 对外主接口: 时间序列对齐
    # ------------------------------------------------------------------

    def calculate_fundamental_series(
        self,
        code: str,
        daily_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        对每个交易日, 找到截止该日最近一期财报, 构建基本面因子时间序列。
        
        参数:
            code: 6位股票代码
            daily_data: 行情 DataFrame, 必须含 'date' 列 (str or datetime)
                        以及 'close' 列 (用于构建 PE / PB)
        
        返回:
            与 daily_data 等长的基本面因子 DataFrame
        """
        dates = pd.to_datetime(daily_data['date'])
        
        # 1. 获取 PIT 对齐的原始财务数据
        raw = self.fetcher.get_pit_series(code, dates)
        
        if raw.empty or len(raw) == 0:
            return pd.DataFrame(index=daily_data.index)

        # 2. 构建因子
        factors = pd.DataFrame(index=daily_data.index)

        # --- 全量保留原始 Baostock 字段作为特征 ---
        for col in self.NUMERIC_COLS:
            factors[col] = pd.to_numeric(raw.get(col), errors='coerce')

        close = pd.to_numeric(daily_data['close'], errors='coerce')

        # --- 衍生 (动态构建 PE/PB & 市值) ---
        # 优先使用 daily_data 原始 pe/pb，若无则利用财务数据推导
        if 'peTTM' in daily_data.columns:
            factors['dynamic_pe'] = pd.to_numeric(daily_data['peTTM'], errors='coerce')
        else:
            factors['dynamic_pe'] = (daily_data['close'] / factors['epsTTM'])

        if 'pbMRQ' in daily_data.columns:
            factors['dynamic_pb'] = pd.to_numeric(daily_data['pbMRQ'], errors='coerce')
        else:
            factors['dynamic_pb'] = np.nan

        factors['inv_pe'] = 1.0 / factors['dynamic_pe'].replace(0, np.nan)
        factors['inv_pb'] = 1.0 / factors['dynamic_pb'].replace(0, np.nan)

        # 市值计算 (单位: 亿)
        factors['market_cap'] = (daily_data['close'] * factors['totalShare'] / 1e8)

        # --- 衍生交叉因子 ---
        factors = self._add_cross_factors(factors)

        # 4. 新增高级复合因子 (PEG, SUE, EAV)
        # PEG = PE / (净利润增长率)
        with np.errstate(divide='ignore', invalid='ignore'):
            # YOYPNI 是百分比，如 20.0 表示 20%
            factors['peg'] = np.where(
                (factors['dynamic_pe'] > 0) & (factors['YOYPNI'] > 0),
                factors['dynamic_pe'] / factors['YOYPNI'],
                np.nan
            )

        # SUE (盈余惊喜) - 简易版: (当前增长 - 过去4期平均增长) / 过去4期标准差
        # 这里因为是 PIT 时间序列，可以使用 rolling
        raw_np_yoy = factors['YOYPNI']
        ma_np = raw_np_yoy.rolling(window=250).mean() # 约一年
        std_np = raw_np_yoy.rolling(window=250).std().replace(0, np.nan)
        factors['sue'] = (raw_np_yoy - ma_np) / std_np

        # EAV (盈利加速度) - YOYPNI 的环比变化 (假设 YOYPNI 是季度数据, 20个交易日约1个月)
        factors['eav'] = factors['YOYPNI'].diff(20) # 约一月的变化加速度

        # 3. 统一清理
        factors = factors.replace([np.inf, -np.inf], np.nan)
        return factors

    # ------------------------------------------------------------------
    # 对外主接口: 最新快照 (选股/实盘)
    # ------------------------------------------------------------------

    def get_latest_fundamental_factors(self, code: str, as_of_date: str = None) -> Dict:
        """
        获取某支股票截止 as_of_date 的最新基本面因子 (dict)。
        
        参数:
            code: 6位股票代码
            as_of_date: 截止日期 'YYYY-MM-DD', None 表示使用今天
        
        返回:
            因子字典, 键为因子名称, 值为 float 或 int
        """
        if as_of_date is None:
            as_of_date = pd.Timestamp.today().strftime('%Y-%m-%d')

        report = self.fetcher.get_pit_report(code, as_of_date)
        if report is None:
            return {}

        factors = {}

        def sf(key): return _safe_float(report.get(key))

        # 1. 全量 Baostock 原始字段
        for col in self.NUMERIC_COLS:
            factors[col] = sf(col)

        # 衍生 (不需要 close, PE/PB 设为 nan)
        factors['dynamic_pe'] = np.nan
        factors['dynamic_pb'] = np.nan
        factors['inv_pe']     = np.nan
        factors['inv_pb']     = np.nan
        factors['peg']        = np.nan
        factors['sue']        = np.nan
        factors['eav']        = np.nan

        # 交叉因子 (基于 dict 版本)
        self._add_cross_factors_dict(factors)

        return factors

    def get_latest_with_price(self, code: str, close: float, as_of_date: str = None) -> Dict:
        """
        获取最新基本面因子, 并利用传入的 close 价格动态构建 PE/PB。
        
        参数:
            code: 6位股票代码
            close: 当前收盘价
            as_of_date: 截止日期
        """
        factors = self.get_latest_fundamental_factors(code, as_of_date)
        if not factors:
            return factors

        eps = factors.get('epsTTM', np.nan)
        bps = factors.get('bps', np.nan)

        if np.isfinite(close) and close > 0:
            if np.isfinite(eps) and eps > 0:
                factors['dynamic_pe'] = close / eps
                factors['inv_pe'] = eps / close
                # 动态 PEG
                np_grow = factors.get('np_yoy', np.nan)
                if np.isfinite(np_grow) and np_grow > 0:
                    factors['peg'] = factors['dynamic_pe'] / np_grow
            if np.isfinite(bps) and bps > 0:
                factors['dynamic_pb'] = close / bps
                factors['inv_pb'] = bps / close

        # 更新交叉因子
        self._add_cross_factors_dict(factors)
        return factors

    # ------------------------------------------------------------------
    # 交叉/衍生因子
    # ------------------------------------------------------------------

    def _add_cross_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """向 DataFrame 追加交叉派生因子"""
        # ROE * 营收成长 -> 成长质量
        roe = factors.get('roeAvg', pd.Series(np.nan, index=factors.index))
        np_yoy = factors.get('YOYPNI', pd.Series(np.nan, index=factors.index))

        factors['roe_x_np_growth']  = roe * (np_yoy / 100).clip(-10, 10)

        # ROE / PB -> 内在价值效率
        pb = factors.get('dynamic_pb', pd.Series(np.nan, index=factors.index))
        with np.errstate(divide='ignore', invalid='ignore'):
            factors['roe_to_pb'] = np.where((pb > 0) & roe.notna(), roe / pb, np.nan)


        return factors

    def _add_cross_factors_dict(self, factors: Dict) -> None:
        """向 dict 追加交叉派生因子 (in-place)"""
        roe    = factors.get('roeAvg', np.nan)
        np_yoy = factors.get('YOYPNI', np.nan)
        pb     = factors.get('dynamic_pb', np.nan)

        def safe_mul(a, b):
            if np.isfinite(a) and np.isfinite(b):
                return a * b
            return np.nan

        def safe_div(a, b):
            if np.isfinite(a) and np.isfinite(b) and b != 0:
                return a / b
            return np.nan

        clip = lambda v, lo=-10, hi=10: max(lo, min(hi, v)) if np.isfinite(v) else np.nan

        factors['roe_x_np_growth']   = safe_mul(roe, clip(np_yoy / 100) if np.isfinite(np_yoy) else np.nan)
        factors['roe_to_pb']         = safe_div(roe, pb)

    # ------------------------------------------------------------------
    # 兼容老接口 (供 advanced_factors.RelativeStrengthFactors 调用)
    # ------------------------------------------------------------------

    def get_stock_info(self, code: str) -> Optional[pd.Series]:
        """兼容旧接口: 返回最新基本面快照 Series"""
        factors = self.get_latest_fundamental_factors(code)
        if not factors:
            return None
        return pd.Series(factors)

    def calculate_all_fundamental_factors(self, code: str) -> Dict:
        """兼容旧接口: 返回最新基本面因子 dict"""
        return self.get_latest_fundamental_factors(code)


# ---------------------------------------------------------------------------
# 市场情绪因子 (MarketSentimentFactors - v2)
# 反映市场多数股票共同表现，而非单股形态
# ---------------------------------------------------------------------------

class MarketSentimentFetcher:
    """
    全市场情绪数据读取器。
    读取由脚本预计算并存储在 stock_meta.db 中的全市场汇总指标。
    """
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        self.meta_db = os.path.join(db_dir, 'stock_meta.db')
        self._cache: Optional[pd.DataFrame] = None

    def _get_all_data(self) -> pd.DataFrame:
        if self._cache is None:
            if not os.path.exists(self.meta_db):
                return pd.DataFrame()
            
            # 使用带 timeout 的连接，避免锁定
            conn = sqlite3.connect(self.meta_db, timeout=30)
            try:
                # 检查表是否存在
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_sentiment'")
                if not cursor.fetchone():
                    return pd.DataFrame()
                
                df = pd.read_sql_query("SELECT * FROM market_sentiment ORDER BY date ASC", conn)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                self._cache = df
            except Exception as e:
                import traceback
                print(f"  [ERROR] 读取市场情绪数据失败: {e}")
                traceback.print_exc()
                self._cache = pd.DataFrame()
            finally:
                conn.close()
        return self._cache

    def get_market_sentiment_series(self, dates: pd.Series) -> pd.DataFrame:
        """为输入的一组日期序列匹配全市场情绪特征"""
        df_all = self._get_all_data()
        
        # 即使数据为空，也要返回正确格式以保证后续逻辑不崩溃
        if df_all is None or df_all.empty:
            # 返回空列但索引对齐的 DataFrame
            return pd.DataFrame(index=dates.index)
        
        # 准备对齐
        dates_dt = pd.to_datetime(dates)
        input_order = pd.DataFrame({'date': dates_dt, 'original_idx': dates.index})
        
        # 合并 (PIT 对齐)
        merged = pd.merge(input_order, df_all, on='date', how='left')
        
        # 移除辅助列
        features = merged.drop(columns=['date', 'original_idx'])
        features.index = dates.index
        
        # 填充缺失值 (某些新日期可能还没预计算)
        return features.ffill().fillna(0)

class MarketSentimentFactors:
    """
    市场情绪因子计算器 (v2 - 全市场汇总版)。
    
    不再基于单股 OHLCV，而是反映全市场 A 股整体的活跃度和广度指标。
    包含比例：上涨家数、大涨家数、涨停家数、市场平均收益、上涨成交量占比、市场广度。
    """

    @staticmethod
    def calculate_all_sentiment_factors(data: pd.DataFrame, db_path: str = DATABASE_PATH) -> pd.DataFrame:
        """
        获取全市场情绪因子。
        
        参数:
            data: 行情 DataFrame，必须包含 'date' 列
            db_path: 数据库路径，用于定位元数据库
        
        返回:
            全市场维度的情绪因子 DataFrame (与其输入日期一一对应)
        """
        if 'date' not in data.columns:
            # 如果没有日期列，尝试使用索引（如果是 DatetimeIndex）
            if isinstance(data.index, pd.DatetimeIndex):
                dates = pd.Series(data.index, index=data.index)
            else:
                return pd.DataFrame(index=data.index)
        else:
            dates = data['date']

        fetcher = MarketSentimentFetcher(db_path)
        return fetcher.get_market_sentiment_series(dates)


# ---------------------------------------------------------------------------
# 便捷函数 (向后兼容)
# ---------------------------------------------------------------------------

def get_fundamental_factors_for_date(code: str, as_of_date: str,
                                      close: float = None,
                                      db_path: str = DATABASE_PATH) -> Dict:
    """便捷函数: 获取某只股票在指定日期的基本面因子"""
    ff = FundamentalFactors(db_path)
    if close is not None and np.isfinite(close) and close > 0:
        return ff.get_latest_with_price(code, close, as_of_date)
    return ff.get_latest_fundamental_factors(code, as_of_date)


if __name__ == '__main__':
    ff = FundamentalFactors()
    print("=== 最新基本面快照 (000001) ===")
    factors = ff.get_latest_fundamental_factors('000001')
    for k, v in factors.items():
        print(f"  {k}: {v}")

    print("\n=== 最新含价格 PE/PB (000001 @ 13.0) ===")
    factors2 = ff.get_latest_with_price('000001', close=13.0)
    for k in ['dynamic_pe', 'dynamic_pb', 'inv_pe', 'inv_pb', 'roe_jq', 'roe_to_pb']:
        print(f"  {k}: {factors2.get(k)}")
