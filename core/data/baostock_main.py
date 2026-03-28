import baostock as bs
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, BrokenExecutor
import time
import pandas as pd
import sqlite3

from .baostock_fetcher import BaostockFetcher
from .baostock_fetcher_methods import (
    fetch_kline_data, fetch_adjust_factor,
    fetch_profit_ability, fetch_growth_ability, 
    fetch_balance_ability, fetch_dupont,
    get_stock_list
)


from config import (
    HISTORY_YEARS, FINANCE_YEARS, WORKERS_NUM, 
    FINANCE_TABLES, SUPPORTED_MARKETS,
    INCREMENTAL_UPDATE, CHECK_LAST_N_DAYS, AUTO_FILL_GAPS,
    SESSION_MAX_STOCKS
)

class BaostockDataManager(BaostockFetcher):
    """Baostock 数据管理器 - 完整实现"""
    
    def update_stock_data(self, code: str, incremental: Optional[bool] = None, 
                          start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        更新单只股票的K线和复权因子数据
        
        参数:
            code: 股票代码
            incremental: 是否增量更新 (None 则使用 config.INCREMENTAL_UPDATE)
            start_date: 强制指定起始日期 (YYYY-MM-DD)
            end_date: 强制指定结束日期 (YYYY-MM-DD)
        """
        try:
            if incremental is None:
                incremental = INCREMENTAL_UPDATE
                
            # 1. 精确到日的增量检查
            now = datetime.now()
            today_str = now.strftime('%Y-%m-%d')
            # 辅助更新时间节点: 17:30 之前认为最新数据是昨天，之后是今天
            trade_target_date = today_str
            if now.hour < 17 or (now.hour == 17 and now.minute < 30):
                trade_target_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')

            check_end_date = trade_target_date if datetime.strptime(end_date,'%Y-%m-%d')>=datetime.strptime(trade_target_date,'%Y-%m-%d') else end_date

            sync_info = self._get_sync_status(code)
            
            if incremental:
                # 如果记录的最后同步日期已经达到或超过了目标交易日，则直接跳过
                if sync_info['daily'] and sync_info['daily'] > check_end_date:
                    return
                # if sync_info['daily'] == today_str: # 兜底逻辑
                #     return

            self._bs_login()
            
            # 确定日期范围
            if end_date and end_date<=trade_target_date:
                calc_end_date = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                calc_end_date = datetime.strptime(trade_target_date, '%Y-%m-%d')


            if incremental:
                last_date = self._get_last_update_date(code)
                if last_date:
                    # 向前回溯 N 天检查完整性
                    calc_start_date = datetime.strptime(last_date, '%Y-%m-%d') - timedelta(days=CHECK_LAST_N_DAYS)
                else:
                    calc_start_date = datetime.now() - timedelta(days=365 * HISTORY_YEARS)
            else:
                calc_start_date = start_date if start_date else datetime.now() - timedelta(days=365 * HISTORY_YEARS)
            
            # 安全检查：解包为字符串
            start_str = calc_start_date.strftime('%Y-%m-%d')
            end_str = calc_end_date.strftime('%Y-%m-%d')
            
            # 如果起始日期晚于结束日期，跳过
            if calc_start_date > calc_end_date:
                return
            
            # 获取K线数据
            kline_df = fetch_kline_data(code, start_str, end_str)
            if not kline_df.empty:
                self._save_kline_data(kline_df)
            else:
                # 如果是空且不是因为日期范围问题，可能需要警告，但 fetch_kline_data 已经打印了错误
                pass
            
            # 获取复权因子
            adjust_df = fetch_adjust_factor(code, start_str, end_str)
            if not adjust_df.empty:
                self._save_adjust_factor(adjust_df)
            
            # 填补数据缺口
            if AUTO_FILL_GAPS:
                self.fill_data_gaps(code)
            
            # 同步成功后记录状态
            if incremental:
                self._update_sync_status(code, 'daily', today_str)
            pass
            
        except Exception as e:
            print(f"✗ {code} 更新失败: {e}")
            raise

    def fill_data_gaps(self, code: str):
        """检查并填补数据缺口"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT MIN(date), MAX(date) FROM daily_data WHERE code = ?", (code,))
        res = cursor.fetchone()
        if not res or not res[0]:
            return
            
        start_date, end_date = res[0], res[1]
        
        # 获取理论交易日
        trade_days = self._get_trade_days(start_date, end_date)
        if not trade_days:
            return
            
        # 获取数据库中已有的交易日
        cursor.execute("SELECT date FROM daily_data WHERE code = ?", (code,))
        existing_days = set(row[0] for row in cursor.fetchall())
        
        # 找出缺失的日期
        missing_days = [d for d in trade_days if d not in existing_days]
        if not missing_days:
            return
            
        print(f"  ℹ {code} 发现 {len(missing_days)} 天数据缺口，正在补全...")
        
        # 补全数据（分段请求以提高效率）
        # 这里为了简单直接全量重新拉取缺失区间的头尾
        kline_df = fetch_kline_data(code, start_date, end_date)
        if not kline_df.empty:
            self._save_kline_data(kline_df)

    def _get_trade_days(self, start_date: str, end_date: str) -> List[str]:
        """获取指定时间段内的交易日列表"""
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        days = []
        while rs.next():
            row = rs.get_row_data()
            if row[1] == '1':  # 1表示交易日
                days.append(row[0])
        return days
    
    def update_finance_data(self, code: str, years: Optional[int] = None, incremental: bool = True):
        """
        更新单只股票的财务数据
        
        参数:
            code: 股票代码
            years: 获取最近几年的数据 (None 则根据 incremental 选择)
            incremental: 增量模式仅检查最近 2 年 (除非指定 years)
        """
        try:
            self._bs_login()
            
            # 2. 季频检测优化: 如果本月已更新且没有历史空缺，则跳过
            now = datetime.now()
            current_month = now.strftime('%Y-%m')
            current_year = now.year
            sync_info = self._get_sync_status(code)
            
            if years is None:
                years = FINANCE_YEARS
            target_start_year = current_year - years + 1

            # 确定时间区间并与上市/退市日期取交集
            ipo_year = self._get_stock_ipo_year(code)
            out_year = self._get_stock_out_year(code)
            
            base_year = target_start_year
            if ipo_year:
                base_year = max(base_year, ipo_year)
                
            end_year = current_year-1
            if out_year:
                end_year = min(end_year, out_year)

            years_to_fetch = []
            
            if incremental:
                # incremental=true则根据上次更新时间判断是否获取最新季度的数据
                if sync_info['finance'] == current_month:
                    return
                # 抓取最近两年的数据（在上市退市区间内）
                years_to_fetch = [y for y in [end_year - 1, end_year] if base_year <= y <= end_year]
            else:
                # incremental=false则准备获取目标区间内的所有数据。取交集
                if base_year <= end_year:
                    years_to_fetch = list(range(base_year, end_year+1))
                # # 3. 核心优化：与数据库中已有的年份时段取补集，避免重复请求
                # existing_min, existing_max = self._get_finance_year_range(code)
                # if existing_min and existing_max:
                #     # 找出目标区间内，数据库中尚未覆盖的年份
                #     full_range = set(range(base_year, end_year + 1))
                #     existing_range = set(range(existing_min, existing_max + 1))
                #     years_to_fetch = sorted(list(full_range - existing_range))
                # else:
                #     years_to_fetch = list(range(base_year, end_year + 1))

            if not years_to_fetch:
                return
            # 并行获取所有财务表
            task_tables = FINANCE_TABLES
            if task_tables:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    for year in years_to_fetch:
                        for quarter in range(1, 5):
                            futures = {}
                            for table in task_tables:
                                # 在每次请求接口前检查数据库中是否已有该年该季度的数据
                                if not self._check_finance_exists(table, code, year, quarter):
                                    fetch_func = globals().get(f'fetch_{table}')
                                    if fetch_func:
                                        futures[executor.submit(fetch_func, code, year, quarter)] = table

                            
                            for future in as_completed(futures):
                                table = futures[future]
                                try:
                                    df = future.result()
                                    if df is not None and not df.empty:
                                        self._save_finance_data(df, table)
                                except Exception as e:
                                    print(f"  ✗ {code} 获取 {table} ({year}Q{quarter}) 失败: {e}")
            # 同步完成后记录月度状态
            self._update_sync_status(code, 'finance', current_month)
            pass
            
        except Exception as e:
            print(f"✗ {code} 财务数据更新失败: {e}")
    
    def init_all_stocks(self, incremental: bool = True, workers: Optional[int] = None, 
                       mode: str = 'all', start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        批量初始化所有股票数据
        
        参数:
            incremental: 是否增量更新
            workers: 并发线程数
            mode: 'all', 'daily', 'finance'
            start_date: 强制起始日期
            end_date: 强制结束日期
        """
        if workers is None:
            workers = WORKERS_NUM
            
        self._bs_login()
        
        # 获取股票列表
        stock_df = get_stock_list()
        if stock_df.empty:
            print("未获取到股票列表")
            return
        
        # 过滤A股
        if 'type' in stock_df.columns:
            stock_df = stock_df[stock_df['type'].isin(['1'])]
        codes = stock_df['code'].tolist()
        
        print(f"开始更新 {len(codes)} 只股票 [Mode: {mode}]...")
        
        worker_func = {
            'all': _update_stock_worker,
            'daily': _update_daily_worker,
            'finance': _update_finance_worker
        }.get(mode, _update_stock_worker)
        
        from tqdm import tqdm
        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = {executor.submit(worker_func, code, incremental, start_date, end_date): code for code in codes}
            with tqdm(total=len(codes), desc=f"同步进度({mode})", unit="只") as pbar:
                _drain_futures(futures, pbar, task_label=mode)
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except (BrokenExecutor, KeyboardInterrupt, OSError):
                pass
        
        print(f"{mode} 数据同步流程结束")

    def update_specific_stocks(self, codes: List[str], incremental: bool = True, 
                               start_date: Optional[str] = None, end_date: Optional[str] = None,
                              workers: Optional[int] = None, mode: str = 'daily'):
        """并行更新指定列表的股票数据"""
        if workers is None:
            workers = WORKERS_NUM
        
        from tqdm import tqdm
        print(f"开始并行更新指定的 {len(codes)} 只股票 [Mode: {mode}] (Workers: {workers})...")
        
        worker_func = {
            'all': _update_stock_worker,
            'daily': _update_daily_worker,
            'finance': _update_finance_worker
        }.get(mode, _update_daily_worker)
        
        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = {executor.submit(worker_func, code, incremental, start_date, end_date): code for code in codes}
            with tqdm(total=len(codes), desc="同步进度", unit="只") as pbar:
                _drain_futures(futures, pbar, task_label=mode)
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except (BrokenExecutor, KeyboardInterrupt, OSError):
                pass
    
    def _get_last_update_date(self, code: str) -> Optional[str]:
        """获取最后更新日期"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(date) FROM daily_data WHERE code = ?", (code,))
        res = cursor.fetchone()
        return res[0] if res and res[0] else None
    
    def _get_finance_year_range(self, code: str) -> Tuple[Optional[int], Optional[int]]:
        """获取财务数据在数据库中的年份范围"""
        cursor = self.conn.cursor()
        # 使用 profit_ability 作为基准表
        try:
            cursor.execute("SELECT MIN(stat_date), MAX(stat_date) FROM finance.profit_ability WHERE code = ?", (code,))
            res = cursor.fetchone()
            if res and res[0] and res[1]:
                min_year = int(res[0][:4])
                max_year = int(res[1][:4])
                return min_year, max_year
        except:
            pass
        return None, None
    
    def _check_finance_exists(self, table: str, code: str, year: int, quarter: int) -> bool:
        """检查财务数据库中是否存在指定年度/季度的数据"""
        cursor = self.conn.cursor()
        # 对应季度的标准报表日期
        quarter_ends = {1: '03-31', 2: '06-30', 3: '09-30', 4: '12-31'}
        target_date = f"{year}-{quarter_ends[quarter]}"
        
        try:
            # stat_date 是大多数财务表的主键之一
            cursor.execute(f"SELECT 1 FROM finance.{table} WHERE code = ? AND stat_date = ?", (code, target_date))
            return cursor.fetchone() is not None
        except:
            return False
    
    def _save_kline_data(self, df: pd.DataFrame):
        """保存K线数据 (批量更新优化)"""
        if df.empty: return
        
        cursor = self.conn.cursor()
        cols = [
            'code', 'date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
            'adjustflag', 'turnover_rate', 'tradestatus', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM',
            'pcfNcfTTM', 'is_st'
        ]
        
        # 准备数据元组
        rows = []
        for _, row in df.iterrows():
            rows.append((
                row['code'], row['date'], row['open'], row['high'], row['low'],
                row['close'], row['preclose'], row['volume'], row['amount'],
                row['adjustflag'], row.get('turnover_rate', 0), row['tradestatus'], row['pctChg'],
                row['peTTM'], row['pbMRQ'], row['psTTM'], row['pcfNcfTTM'], row.get('is_st', 0)
            ))
            
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)
        sql = f"INSERT OR REPLACE INTO daily_data ({col_names}) VALUES ({placeholders})"
        
        cursor.executemany(sql, rows)
        self.safe_commit()
    
    def _save_adjust_factor(self, df: pd.DataFrame):
        """保存复权因子 (批量更新优化)"""
        if df.empty: return
        cursor = self.conn.cursor()
        rows = [(row['code'], row['date'], row['fore_adjust_factor'], row['back_adjust_factor']) 
                for _, row in df.iterrows()]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO adjust_factor 
            (code, date, fore_adjust_factor, back_adjust_factor)
            VALUES (?, ?, ?, ?)
        ''', rows)
        self.safe_commit()
    
    # 缓存表的列名信息，避免 PRAGMA 重复执行
    _table_cols_cache = {}

    def _save_finance_data(self, df: pd.DataFrame, table_name: str):
        """保存财务数据到指定表 (批量更新优化)"""
        if df.empty:
            return
        
        cursor = self.conn.cursor()
        
        # 缓存机制：减少 PRAGMA 查询
        if table_name not in self._table_cols_cache:
            cursor.execute(f"PRAGMA finance.table_info({table_name})")
            self._table_cols_cache[table_name] = [col[1] for col in cursor.fetchall()]
        
        valid_cols = self._table_cols_cache[table_name]
        columns = [str(col) for col in df.columns if col in valid_cols]
        
        if not columns:
            return
            
        placeholders = ', '.join(['?'] * len(columns))
        col_names = ', '.join(columns)
        
        rows = []
        for _, row in df.iterrows():
            rows.append(tuple(row[col] for col in columns))
            
        try:
            cursor.executemany(f'''
                INSERT OR REPLACE INTO finance.{table_name} ({col_names})
                VALUES ({placeholders})
            ''', rows)
            self.safe_commit()
        except Exception as e:
            print(f"  ✗ 批量写入表 {table_name} 失败: {e}")
    
    def get_adjusted_kline(self, code: str, start_date: str, end_date: str, 
                          adjust_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取动态前复权K线数据
        
        参数:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust_date: 复权基准日（None则使用end_date）
        
        返回:
            DataFrame with adjusted prices
        """
        if adjust_date is None:
            adjust_date = end_date
        
        # 获取原始K线
        query = '''
            SELECT k.*, a.fore_adjust_factor
            FROM daily_data k
            LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
            WHERE k.code = ? AND k.date >= ? AND k.date <= ?
            ORDER BY k.date
        '''
        df = pd.read_sql_query(query, self.conn, params=(code, start_date, end_date))
        
        if df.empty:
            return df
        
        # 获取基准日的复权因子
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT fore_adjust_factor FROM adjust_factor 
            WHERE code = ? AND date <= ? 
            ORDER BY date DESC LIMIT 1
        ''', (code, adjust_date))
        result = cursor.fetchone()
        
        if not result or result[0] is None:
            print(f"警告: {code} 在 {adjust_date} 没有复权因子，返回不复权数据")
            return df
        
        base_factor = result[0]
        
        # 动态前复权计算
        df['fore_adjust_factor'] = df['fore_adjust_factor'].fillna(1.0)
        df['adj_factor_ratio'] = df['fore_adjust_factor'] / base_factor
        
        # 复权价格 = 原始价格 * (当日复权因子 / 基准日复权因子)
        for col in ['open', 'high', 'low', 'close', 'preclose']:
            df[f'adj_{col}'] = df[col] * df['adj_factor_ratio']
        
        # 成交量不需要复权
        df['adj_volume'] = df['volume']
        df['adj_amount'] = df['amount']
        
        return df
    
    def get_stock_list_from_db(self, markets: Optional[List[str]] = None) -> pd.DataFrame:
        """
        从数据库获取股票列表
        
        参数:
            markets: 市场列表 ['sh', 'sz', 'bj']
        
        返回:
            DataFrame with stock info
        """
        query = "SELECT DISTINCT code FROM daily_data"
        df = pd.read_sql_query(query, self.conn)
        
        if markets:
            prefixes = []
            for market in markets:
                if market in SUPPORTED_MARKETS:
                    prefixes.extend(SUPPORTED_MARKETS[market]['prefixes'])
            
            if prefixes:
                df = df[df['code'].str.startswith(tuple(prefixes))]
        
        return df


# 单个任务超时阈值（秒）。单只股票超过此时间未完成则跳过，防止卡死
try:
    from config import TASK_TIMEOUT_SECONDS
except ImportError:
    TASK_TIMEOUT_SECONDS = 120

def _drain_futures(futures: dict, pbar, task_label: str = ""):
    """
    带超时的 future 结果收集器。
    逐一等待每个 future，超过 TASK_TIMEOUT_SECONDS 则打印警告并跳过，
    保证进度条始终可以走到 100%。
    """
    from concurrent.futures import TimeoutError as FuturesTimeout
    pending = set(futures.keys())
    while pending:
        # as_completed 本身支持 timeout，超时后抛 TimeoutError
        # 但我们用手动轮询更可控
        done_now = []
        for future in list(pending):
            if future.done():
                done_now.append(future)
        
        if done_now:
            for future in done_now:
                code = futures[future]
                try:
                    future.result(timeout=0)
                except Exception as e:
                    print(f"\n进程执行失败 [{task_label}] {code}: {e}")
                finally:
                    pbar.update(1)
                    pending.discard(future)
        else:
            # 没有已完成的，等待一批（带超时）
            try:
                for future in as_completed(list(pending), timeout=TASK_TIMEOUT_SECONDS):
                    code = futures[future]
                    try:
                        future.result(timeout=0)
                    except Exception as e:
                        print(f"\n进程执行失败 [{task_label}] {code}: {e}")
                    finally:
                        pbar.update(1)
                        pending.discard(future)
                    break  # 每次只取一个，重新评估 pending
            except TimeoutError:
                # 超时：找出还没完成的任务，逐一打印警告并跳过
                for future in list(pending):
                    if not future.done():
                        code = futures[future]
                        print(f"\n⚠ [{task_label}] {code} 超过 {TASK_TIMEOUT_SECONDS}s 未响应，已跳过")
                        future.cancel()
                        pbar.update(1)
                        pending.discard(future)


# 全局计数器，用于跟踪当前进程处理的任务数
_process_task_count = 0

def _update_stock_worker(code, incremental, start_date=None, end_date=None):
    """多进程工作函数: 同时更新行情和财务"""
    _execute_worker_task(code, incremental, update_daily=True, update_finance=True, 
                         start_date=start_date, end_date=end_date)

def _update_daily_worker(code, incremental, start_date=None, end_date=None):
    """多进程工作函数: 仅更新行情"""
    _execute_worker_task(code, incremental, update_daily=True, update_finance=False,
                         start_date=start_date, end_date=end_date)

def _update_finance_worker(code, incremental, start_date=None, end_date=None):
    """多进程工作函数: 仅更新财务"""
    _execute_worker_task(code, incremental, update_daily=False, update_finance=True,
                         start_date=start_date, end_date=end_date)

def _execute_worker_task(code, incremental, update_daily=True, update_finance=True,
                         start_date=None, end_date=None):
    """执行具体任务的通用内部函数"""
    global _process_task_count
    import os
    import time
    pid = os.getpid()
    
    manager = BaostockDataManager()
    try:
        if update_daily:
            manager.update_stock_data(code, incremental, start_date=start_date, end_date=end_date)
        if update_finance:
            manager.update_finance_data(code, incremental=incremental)
            
        _process_task_count += 1
        
        # 核心优化：当当前会话复用次数达到阈值时，主动注销重连
        if _process_task_count >= SESSION_MAX_STOCKS:
            print(f"[PID {pid}] 会话复用达到阈值 ({SESSION_MAX_STOCKS})，正在重置连接...")
            manager.logout()
            _process_task_count = 0
            
    except Exception as e:
        print(f"[PID {pid}] 处理 {code} 时出错: {e}")
    finally:
        manager.close()
