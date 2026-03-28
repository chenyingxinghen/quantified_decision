import pandas as pd
import baostock as bs
import threading
import time
import os
from typing import Optional, List, Any
from config import MARKET_PREFIXES, ADJUST_FLAG, REQUEST_INTERVAL


class CachedResultSet:
    """模拟 Baostock ResultSet 的对象，预抓取所有数据以保证线程安全"""
    def __init__(self, rs):
        self.error_code = getattr(rs, 'error_code', '0')
        self.error_msg = getattr(rs, 'error_msg', '')
        self.fields = getattr(rs, 'fields', [])
        self.data = []
        while rs.next():
            self.data.append(rs.get_row_data())
        self._pos = 0

    def next(self):
        if self._pos < len(self.data):
            self._pos += 1
            return True
        return False

    def get_row_data(self):
        if 0 < self._pos <= len(self.data):
            return self.data[self._pos - 1]
        return []


def _bs_query(method_name: str, **kwargs) -> Any:
    """
    通用 Baostock 查询助手，集成锁、重试和请求间隔
    """
    max_retries = 3
    last_error = ""
    method = getattr(bs, method_name)
    
    for attempt in range(max_retries):
        try:
            if REQUEST_INTERVAL > 0:
                time.sleep(REQUEST_INTERVAL)
                
            rs = method(**kwargs)
            
            if rs is None:
                continue

            if rs.error_code != '0':
                last_error = rs.error_msg
                
                if "用户未登录" in last_error or "you don't login" in last_error.lower():
                    # 核心修复：如果发现未登录，尝试重新登录并重试
                    from .baostock_fetcher import BaostockFetcher
                    import os
                    print(f"  [PID {os.getpid()}] ⚠ 检测到会话失效，正在重新登录...")
                    if BaostockFetcher._bs_login():
                        continue # 重新循环，执行 method(**kwargs)
                
                if "接收数据异常" in last_error or "网络接收错误" in last_error or "10001001" in last_error:
                    print(f"  ⚠ {method_name} 第 {attempt+1} 次失败: {last_error}，正在重试...")
                    continue
                else:
                    print(f"✗ {method_name} 失败: {last_error}")
                    return None
            
            # 预抓取所有数据
            return CachedResultSet(rs)
                
        except Exception as e:
            last_error = str(e)
            print(f"  ⚠ {method_name} 第 {attempt+1} 次异常: {last_error}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
            
    print(f"✗ {method_name} 最终失败: {last_error}")
    return None


def _to_bs_symbol(code: str) -> str:
    """根据配置转换股票代码为 baostock 格式 (sh.xxxxxx, sz.xxxxxx, bj.xxxxxx)"""
    # 优先匹配主板/科创板 (sh)
    if code.startswith(('60', '68')):
        return f"sh.{code}"
    # 匹配深市 (sz)
    if code.startswith(('00', '30')):
        return f"sz.{code}"
    # 匹配北交所 (bj)
    # 北交所前缀通常为 8, 4, 9
    if code.startswith(MARKET_PREFIXES['bj']):
        return f"bj.{code}"
    
    return code


def _from_bs_symbol(bs_code: str) -> str:
    """从 baostock 格式转换为标准代码"""
    if '.' in bs_code:
        return bs_code.split('.')[1]
    return bs_code


def fetch_kline_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取K线数据"""
    bs_code = _to_bs_symbol(code)
    rs = _bs_query("query_history_k_data_plus",
                  code=bs_code,
                  fields="date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                  start_date=start_date, end_date=end_date,
                  frequency="d", adjustflag=ADJUST_FLAG)
    if not rs: return pd.DataFrame()
    
    data = []
    while rs.next(): data.append(rs.get_row_data())
    if not data: return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=rs.fields)
    for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'tradestatus' in df.columns: df['tradestatus'] = pd.to_numeric(df['tradestatus'], errors='coerce').fillna(0).astype(int)
    if 'isST' in df.columns: df['isST'] = pd.to_numeric(df['isST'], errors='coerce').fillna(0).astype(int)
    if 'code' in df.columns: df['code'] = df['code'].apply(_from_bs_symbol)
    return df.rename(columns={'turn': 'turnover_rate', 'isST': 'is_st'})


def fetch_adjust_factor(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取复权因子"""
    bs_code = _to_bs_symbol(code)
    rs = _bs_query("query_adjust_factor", code=bs_code, start_date=start_date, end_date=end_date)
    if not rs: return pd.DataFrame()
    
    data = []
    while rs.next(): data.append(rs.get_row_data())
    if not data: return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=rs.fields)
    for col in ['foreAdjustFactor', 'backAdjustFactor']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.rename(columns={'foreAdjustFactor': 'fore_adjust_factor', 'backAdjustFactor': 'back_adjust_factor', 'dividOperateDate': 'date'})
    if 'code' in df.columns: df['code'] = df['code'].apply(_from_bs_symbol)
    
    required_cols = ['date', 'code', 'fore_adjust_factor', 'back_adjust_factor']
    for col in required_cols:
        if col not in df.columns: df[col] = None if col != 'date' else start_date
    return df[required_cols]


def _fetch_finance_data(method_name: str, code: str, year: int, quarter: int) -> pd.DataFrame:
    """通用财务数据获取助手"""
    bs_code = _to_bs_symbol(code)
    rs = _bs_query(method_name, code=bs_code, year=year, quarter=quarter)
    if not rs: return pd.DataFrame()
    
    data = []
    while rs.next(): data.append(rs.get_row_data())
    if not data: return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=rs.fields)
    if 'code' in df.columns: df['code'] = df['code'].apply(_from_bs_symbol)
    return df.rename(columns={'pubDate': 'pub_date', 'statDate': 'stat_date'})


def fetch_profit_ability(code: str, year: int, quarter: int) -> pd.DataFrame:
    """获取盈利能力数据"""
    return _fetch_finance_data("query_profit_data", code, year, quarter)


def fetch_operation_ability(code: str, year: int, quarter: int) -> pd.DataFrame:
    """获取营运能力数据"""
    return _fetch_finance_data("query_operation_data", code, year, quarter)


def fetch_growth_ability(code: str, year: int, quarter: int) -> pd.DataFrame:
    """获取成长能力数据"""
    return _fetch_finance_data("query_growth_data", code, year, quarter)


def fetch_balance_ability(code: str, year: int, quarter: int) -> pd.DataFrame:
    """获取偿债能力数据"""
    return _fetch_finance_data("query_balance_data", code, year, quarter)


def fetch_cash_flow(code: str, year: int, quarter: int) -> pd.DataFrame:
    """获取现金流量数据"""
    return _fetch_finance_data("query_cash_flow_data", code, year, quarter)


def fetch_dupont(code: str, year: int, quarter: int) -> pd.DataFrame:
    """获取杜邦指数数据"""
    return _fetch_finance_data("query_dupont_data", code, year, quarter)


def fetch_performance_forecast(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取业绩预告数据"""
    bs_code = _to_bs_symbol(code)
    rs = _bs_query("query_forecast_report", code=bs_code, start_date=start_date, end_date=end_date)
    if not rs: return pd.DataFrame()
    data = []
    while rs.next(): data.append(rs.get_row_data())
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data, columns=rs.fields)
    if 'code' in df.columns: df['code'] = df['code'].apply(_from_bs_symbol)
    return df.rename(columns={'pubDate': 'pub_date', 'statDate': 'stat_date'})


def get_stock_list(markets:Optional[List[str]] = None) -> pd.DataFrame:
    """获取所有股票列表 (带数据库缓存)"""
    from .baostock_fetcher import BaostockFetcher
    from datetime import datetime
    
    fetcher = BaostockFetcher()
    if markets:
        from .baostock_main import BaostockDataManager
        manager=BaostockDataManager()
        return manager.get_stock_list_from_db(markets)
    try:
        # 1. 尝试从数据库获取
        db_df = fetcher._get_stock_basic_from_db()
        if not db_df.empty:
            last_update_str = db_df['update_time'].max()
            if last_update_str:
                last_update = datetime.strptime(last_update_str, '%Y-%m-%d %H:%M:%S')
                # 如果是最近一月内更新的，直接返回
                if (datetime.now() - last_update).total_seconds() < 30*24 * 3600:
                    print(f"  ℹ 使用缓存的股票列表 (最后更新: {last_update_str})")
                    return db_df.drop(columns=['update_time'])
    except Exception as e:
        print(f"  ⚠ 数据库读取股票列表失败: {e}")
        db_df = pd.DataFrame()

    # 2. 从 Baostock 获取
    print("  🌐 正在从 Baostock 获取最新股票列表...")
    rs = _bs_query("query_stock_basic")
    if not rs: 
        if not db_df.empty:
            print("  ⚠ 从 Baostock 获取失败，使用数据库旧数据兜底")
            return db_df.drop(columns=['update_time'])
        return pd.DataFrame()
        
    data = []
    while rs.next(): data.append(rs.get_row_data())
    if not data: 
        if not db_df.empty: return db_df.drop(columns=['update_time'])
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=rs.fields)
    df['code'] = df['code'].apply(_from_bs_symbol)
    
    # 3. 存储到数据库
    try:
        fetcher._save_stock_basic_to_db(df)
    except Exception as e:
        print(f"  ⚠ 缓存股票列表失败: {e}")
    finally:
        fetcher.close()
        
    return df


def fetch_stock_industry(code: Optional[str] = None, date: Optional[str] = None) -> pd.DataFrame:
    """获取股票行业分类信息"""
    bs_code = _to_bs_symbol(code) if code else ""
    rs = _bs_query("query_stock_industry", code=bs_code, date=date)
    if not rs:
        return pd.DataFrame()

    data = []
    while rs.next():
        data.append(rs.get_row_data())
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=rs.fields)
    if 'code' in df.columns:
        df['code'] = df['code'].apply(_from_bs_symbol)

    # 重命名列以符合项目规范
    return df.rename(columns={
        'updateDate': 'update_date',
        'code_name': 'stock_name',
        'industryClassification': 'industry_classification'
    })
