import pandas as pd
import baostock as bs
import threading
import time
import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Any
from config import MARKET_PREFIXES, ADJUST_FLAG, REQUEST_INTERVAL, API_DAILY_QUOTA

class QuotaExceededError(Exception):
    pass

def _register_api_call(n: int = 1, gate: bool = True):
    """
    登记一次（或 n 次）Baostock API 调用到每日配额表。

    - gate=True  : 原子地「检查上限 + 计数」，若已超限则抛 QuotaExceededError（用于正式数据请求前的拦截）。
    - gate=False : 仅计数（用于 login / 交易日查询等前置调用，永不拦截）。

    计数在单条 SQLite UPDATE 内原子完成，避免原先「先 SELECT 再 UPDATE」的非原子竞态。
    配额表缺失 / 锁等待等异常不再静默吞掉，改为打印告警，防止配额保护在出错时悄悄失效。
    """
    from config.baostock_config import META_DB_PATH
    max_quota = API_DAILY_QUOTA
    today = datetime.now().strftime('%Y-%m-%d')
    last_err = None
    # 短暂重试以应对瞬时 database is locked（如另一进程正在写 sync_status）；非跨进程互斥锁
    for _ in range(3):
        try:
            with sqlite3.connect(META_DB_PATH, timeout=5.0) as conn:
                cur = conn.cursor()
                cur.execute("PRAGMA busy_timeout=5000")
                # 确保当日行存在
                cur.execute("INSERT OR IGNORE INTO api_quota(date, count) VALUES (?, 0)", (today,))
                if gate:
                    # 仅当 count + n <= max_quota 时才累加；否则不累加并抛出
                    cur.execute(
                        "UPDATE api_quota SET count = count + ? WHERE date = ? AND count + ? <= ?",
                        (n, today, n, max_quota),
                    )
                    if cur.rowcount == 0:
                        cur.execute("SELECT count FROM api_quota WHERE date = ?", (today,))
                        row = cur.fetchone()
                        cnt = row[0] if row else '?'
                        raise QuotaExceededError(f"Daily Baostock API quota exceeded ({cnt}/{max_quota})")
                else:
                    # 仅计数：到顶后不再累加，但不拦截
                    cur.execute(
                        "UPDATE api_quota SET count = count + ? WHERE date = ? AND count + ? <= ?",
                        (n, today, n, max_quota),
                    )
                conn.commit()
            return
        except QuotaExceededError:
            raise
        except sqlite3.OperationalError as e:
            last_err = e
            time.sleep(0.1)
            continue
    # 重试后仍失败（表缺失 / 持续锁等待等）：记录告警，避免配额保护在出错时悄悄失效
    print(f"[quota] 警告: 配额登记失败 ({last_err})")

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

    # 每个逻辑请求在正式发起前登记一次配额：原子地检查上限并计数，超限即抛异常拦截。
    # 放在重试循环之外 => 每个逻辑请求只计一次（含其内部重试），杜绝原先每次重试重复计数的问题。
    _register_api_call(1, gate=True)

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
                    try:
                        from .baostock_fetcher import BaostockFetcher
                        BaostockFetcher._bs_login()
                    except Exception:
                        pass
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"✗ {method_name} 失败: {last_error}")
                    return None
            
            # 预抓取所有数据
            return CachedResultSet(rs)
                
        except QuotaExceededError:
            raise
        except Exception as e:
            last_error = str(e)
            print(f"  ⚠ {method_name} 第 {attempt+1} 次异常: {last_error}")
            # WinError 10057 / 10054 等 socket 断连错误，需要重新登录后再重试
            socket_errors = ("10057", "10054", "10053", "套接字", "socket", "WinError")
            if any(kw in last_error for kw in socket_errors):
                try:
                    from .baostock_fetcher import BaostockFetcher
                    print(f"  [PID {os.getpid()}] ⚠ 检测到 socket 断连，正在重新登录...")
                    BaostockFetcher._bs_login()
                except Exception as login_err:
                    print(f"  ⚠ 重新登录失败: {login_err}")
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
    # query_forecast_report 的字段名和其它财务接口不一样：没有 pubDate/statDate，
    # 而是 profitForcastExpPubDate（预告披露日）/ profitForcastExpStatDate（报告期）。
    # 老代码只按 pubDate/statDate 改名 => 永远改不到，下游 dropna(subset=['pub_date'])
    # 把每只票都清空，落库 0 行却一声不响（E10「抓不到预告」的真因）。
    df = df.rename(columns={'pubDate': 'pub_date', 'statDate': 'stat_date'})
    if 'pub_date' not in df.columns and 'profitForcastExpPubDate' in df.columns:
        df['pub_date'] = df['profitForcastExpPubDate']
    if 'stat_date' not in df.columns and 'profitForcastExpStatDate' in df.columns:
        df['stat_date'] = df['profitForcastExpStatDate']
    return df


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
