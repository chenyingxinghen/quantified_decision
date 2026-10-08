import baostock as bs
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, BrokenExecutor
import time
import pandas as pd
import numpy as np
import sqlite3

from .baostock_fetcher import BaostockFetcher
from .baostock_fetcher_methods import (
    fetch_kline_data, fetch_adjust_factor,
    fetch_profit_ability, fetch_growth_ability, 
    fetch_balance_ability, fetch_dupont,
    get_stock_list, _register_api_call, quota_remaining
)


from config import (
    HISTORY_YEARS, FINANCE_YEARS, WORKERS_NUM,
    FINANCE_TABLES, SUPPORTED_MARKETS,
    INCREMENTAL_UPDATE, CHECK_LAST_N_DAYS, AUTO_FILL_GAPS,
    SESSION_MAX_STOCKS,
)

# 这几个超时常量直接从源头模块导入，不走 `config/__init__.py` 的 __all__ 白名单 ——
# 2026-09-24 早上 8 点实盘选股任务挂掉就是因为白名单没同步：
# ImportError: cannot import name 'TASK_HARD_TIMEOUT_SECONDS' from 'config'。
from config.baostock_config import (
    TASK_HARD_TIMEOUT_SECONDS, TASK_MAX_STALLS, MAX_POOL_RESTARTS,
    BAOSTOCK_SOCKET_TIMEOUT,
)

# 各季度报表对应的报告期截止日（stat_date 的月-日部分）。
QUARTER_ENDS = {1: '03-31', 2: '06-30', 3: '09-30', 4: '12-31'}

# low_freq_sync 表里复权因子这一类的 kind 名。同表可容纳其他低频数据集
# （变更时点不定、且不该因为"没拿到数据"就每晚重跑的东西）。
ADJUST_FACTOR_KIND = 'adjust_factor'


def quarter_published(year: int, quarter: int, today: Optional[str] = None) -> bool:
    """该季度现在**有没有可能**已经披露。

    规则只有一条：报告期没过完，财报不可能存在。2026-10-08 时 2026Q3
    （报告期 09-30 已过）值得问，2026Q4（报告期 12-31）答案是空——但每个
    交易日仍要为全市场 5000+ 只各发 4 次请求，纯粹烧配额。

    这只是**下界**，不代表已披露（三季报要到 10 月下旬才陆续出）。真正的
    "有没有数据"仍由 ``_check_finance_exists`` 与接口返回决定；这里只是
    不把明显不可能存在的季度发出去。
    """
    if today is None:
        today = datetime.now().strftime('%Y-%m-%d')
    return today > f"{year}-{QUARTER_ENDS[quarter]}"


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

            # end_date 允许为 None（update_multiple_stocks / 手动补数时常不传）；
            # 旧代码直接 strptime(end_date) 会抛 TypeError，让整只股票更新失败。
            if end_date and datetime.strptime(end_date, '%Y-%m-%d') >= datetime.strptime(trade_target_date, '%Y-%m-%d'):
                check_end_date = end_date
            else:
                check_end_date = trade_target_date

            sync_info = self._get_sync_status(code)
            
            if incremental:
                # 如果记录的最后同步日期已经达到或超过了目标交易日，则直接跳过
                if sync_info['daily'] and sync_info['daily'] > check_end_date:
                    print(f'sync_info:{sync_info['daily']} > check_end_date:{check_end_date}')
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
                calc_start_date = start_date if start_date else datetime.strptime('2009-03-03','%Y-%m-%d')
            
            # 安全检查：解包为字符串
            start_str = calc_start_date.strftime('%Y-%m-%d')
            end_str = calc_end_date.strftime('%Y-%m-%d')
            
            # 如果起始日期晚于结束日期，跳过
            if calc_start_date > calc_end_date:
                print(f'calc_start_date:{calc_start_date} > cal_end_date:{calc_end_date}')
                return
            
            # 获取K线数据
            kline_df = fetch_kline_data(code, start_str, end_str)
            kline_written = False
            if not kline_df.empty:
                self._save_kline_data(kline_df)
                kline_written = True
            else:
                # 如果是空且不是因为日期范围问题，可能需要警告，但 fetch_kline_data 已经打印了错误
                pass
            
            # 复权因子不再在每日增量里全量重抓（每只 1 次 × 全市场 ≈ 5480 次/天，
            # 99% 与昨日完全相同）。改由 `update_adjust_factor_data` 低频刷新，
            # 它自带独立的月度标记（见该方法的 docstring，别改回挂在财务 stamp 上）。
            # adjust_factor 表保留供调试/可视化。
            pass
            # 填补数据缺口
            if AUTO_FILL_GAPS:
                self.fill_data_gaps(code)
            
            # 仅在实际写入、且返回的数据真推进到了目标交易日时才记录同步日。
            # 早先版本：只要返回非空历史就 stamp today —— 长期停牌股每次都会返回一段
            # “截至停牌日为止”的非空历史，于是被误标成“已同步到今天”（verify 里即 sync_status 谎言），
            # 还在每个夜晚为每只死股白烧一次抓取。停牌/退市股到不了 target，改为不 stamp：
            # 保留旧状态，让它持续可重试、也不伪造同步。若 baostock 返回空（拉黑/网络错/真无交易日），
            # 同样不 stamp —— 这正是 09-04 整片拉取失败时缺口能被下一次增量自动回填的前提。
            if incremental and kline_written \
                    and str(kline_df['date'].max()) >= check_end_date:
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
        # 交易日查询也是一次 API 调用，纳入每日配额计数（仅计数，不拦截）
        try:
            _register_api_call(1, gate=False)
        except Exception:
            pass
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
                
            end_year = current_year
            if out_year:
                end_year = min(end_year, out_year)

            years_to_fetch = []
            
            if incremental:
                # incremental=true则根据上次更新时间判断是否获取最新季度的数据
                if sync_info['finance'] == current_month:
                    return
                # 复权因子低频刷新：自带独立月度标记，频率与这里是否真的存进
                # 财务数据无关（见 update_adjust_factor_data 的 docstring）。
                self.update_adjust_factor_data(code)
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
            saved_any = False
            if task_tables:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    for year in years_to_fetch:
                        for quarter in range(1, 5):
                            # 报告期还没过完的季度一定没有数据，不发请求。
                            # 见 quarter_published 的说明：这条护栏在月末/年初
                            # 能砍掉每只股票一半的无效调用。
                            if not quarter_published(year, quarter):
                                continue
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
                                        saved_any = True
                                except Exception as e:
                                    print(f"  ✗ {code} 获取 {table} ({year}Q{quarter}) 失败: {e}")
            # 仅在实际写入财务数据时才记录月度状态，避免“假成功”状态掩盖缺口
            if saved_any:
                self._update_sync_status(code, 'finance', current_month)
            pass
            
        except Exception as e:
            print(f"✗ {code} 财务数据更新失败: {e}")
    
    def update_adjust_factor_data(self, code: str, end_date: Optional[str] = None):
        """低频刷新单只股票的复权因子（每月一次）。

        前复权因子历史会随除权除息整体重算，所以单只内必须全量重取；但频率已从
        每日降到每月，把每天的 ~5480 次全量请求压成每月一次。

        ⚠ 这里**不能**靠 `last_finance_sync` 来节制：那个 stamp 的语义是「财务
        数据已入库」，只在真写进行时才落。三季报/年报披露前财务接口天天返回空，
        于是它整月都不 stamp，本方法就会每晚陪跑一次全历史重抓。所以复权因子
        走自己的标记表 `low_freq_sync`，语义是「本月已经检查过」，与这次拿到
        多少数据无关 —— 只要请求正常返回（空返回即已退市等结构性缺失，本月不会
        变）就打标；抛异常则不打标，明晚重试。
        """
        current_month = datetime.now().strftime('%Y-%m')
        if self.get_low_freq_marker(code, ADJUST_FACTOR_KIND) == current_month:
            return
        try:
            if end_date is None:
                now = datetime.now()
                trade_target_date = now.strftime('%Y-%m-%d')
                if now.hour < 17 or (now.hour == 17 and now.minute < 30):
                    trade_target_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
                end_date = trade_target_date

            self._bs_login()
            full_start = '2009-03-03'
            adjust_df = fetch_adjust_factor(code, full_start, end_date)
            if not adjust_df.empty:
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM adjust_factor WHERE code = ?", (code,))
                self._save_adjust_factor(adjust_df)
            # 请求正常返回了才算「检查过」——放在空的 if 外面，退市股也要打标，
            # 否则它们同样会每晚重抓（docstring 里说的正是这个坑）。
            self.set_low_freq_marker(code, ADJUST_FACTOR_KIND, current_month)
        except Exception as e:
            print(f"✗ {code} 复权因子刷新失败: {e}")

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
            self.logout()  # 释放主进程会话锁，避免常驻占用导致 worker 拿不到锁
            return
        
        # 过滤A股
        if 'type' in stock_df.columns:
            stock_df = stock_df[stock_df['type'].isin(['1'])]
        codes = stock_df['code'].tolist()
        
        # 主进程仅用于取列表，取完即注销并释放跨进程会话锁；
        # 真正的 API 请求交给 worker 进程各自登录，确保任意时刻全机器仅 1 个 baostock 会话。
        self.logout()
        
        print(f"开始更新 {len(codes)} 只股票 [Mode: {mode}]...")
        worker_func = {
            'all': _update_stock_worker,
            'daily': _update_daily_worker,
            'finance': _update_finance_worker
        }.get(mode, _update_stock_worker)

        from tqdm import tqdm
        import sys
        is_atty = sys.stdout.isatty()

        pbar = tqdm(total=len(codes), desc=f"同步进度({mode})", unit="只", disable=not is_atty)
        pending_codes = list(codes)
        try:
            for round_no in range(1, MAX_POOL_RESTARTS + 2):
                if not pending_codes:
                    break
                executor = ProcessPoolExecutor(max_workers=workers)
                try:
                    futures = {executor.submit(worker_func, c, incremental,
                                               start_date, end_date): c for c in pending_codes}
                    remaining, killed, abort_reason = _drain_futures(
                        futures, pbar, task_label=mode, executor=executor)
                finally:
                    try:
                        for proc in executor._processes.values():
                            try: proc.terminate()
                            except Exception: pass
                        executor.shutdown(wait=False, cancel_futures=True)
                    except (BrokenExecutor, KeyboardInterrupt, OSError):
                        pass

                if not remaining:
                    break
                if abort_reason or not killed:
                    # 配额打满 / 整批系统性故障（abort_reason）或据实无进展（killed=False）：
                    # 重建进程池只会原样再失败一遍，直接收尾。
                    print(f"[{mode}] 第 {round_no} 轮中止（{abort_reason or '系统性故障'}），"
                          f"剩余 {len(remaining)} 只不再续跑；已落库的部分是 checkpoint，"
                          f"下次增量会接着做")
                    break
                if round_no > MAX_POOL_RESTARTS:
                    print(f"[{mode}] 已重建进程池 {MAX_POOL_RESTARTS} 次，"
                          f"仍有 {len(remaining)} 只未处理，停止续跑")
                    break
                print(f"♻ [{mode}] 第 {round_no} 轮结束，仍有 {len(remaining)} 只未处理，"
                      f"重建进程池续跑...")
                pending_codes = remaining
        finally:
            pbar.close()

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
                remaining, killed, abort_reason = _drain_futures(
                    futures, pbar, task_label=mode, executor=executor)
            if remaining:
                print(f"[{mode}] 有 {len(remaining)} 只未处理"
                      + (f"（{abort_reason}）" if abort_reason else ""))
        finally:
            try:
                for proc in executor._processes.values():
                    try: proc.terminate()
                    except Exception: pass
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
        target_date = f"{year}-{QUARTER_ENDS[quarter]}"
        
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
        获取动态前复权K线数据（用 preclose/close 反推，覆盖 100%，不再依赖 adjust_factor 表）

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
        
        # 获取原始K线（含 preclose，作为完整前复权的唯一来源）
        query = '''
            SELECT k.date, k.open, k.high, k.low, k.close, k.preclose,
                   k.volume, k.amount
            FROM daily_data k
            WHERE k.code = ? AND k.date >= ? AND k.date <= ?
            ORDER BY k.date
        '''
        df = pd.read_sql_query(query, self.conn, params=(code, start_date, end_date))
        
        if df.empty:
            return df
        
        # 从 preclose/close 反推完整前向复权因子（锚定在窗口末日 = end_date）
        from core.factors.train_ml_model import compute_forward_adjust_factor
        f = compute_forward_adjust_factor(
            df['close'].to_numpy(dtype=np.float64),
            df['preclose'].to_numpy(dtype=np.float64),
        )
        # 以 adjust_date 的原始价为锚，把整段重新缩放（forward-adjust 的平移是统一比例缩放）
        dates = list(df['date'])
        base_idx = dates.index(adjust_date) if adjust_date in dates else len(df) - 1
        f_base = f[base_idx]
        if f_base == 0 or pd.isna(f_base):
            f_base = 1.0
        scale = f / f_base
        
        for col in ['open', 'high', 'low', 'close', 'preclose']:
            raw = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float64)
            df[col] = (raw * scale).astype(np.float32)
            df[f'adj_{col}'] = df[col]
        
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
                else:
                    # 尝试匹配 exchange code (如 'sh' 匹配 'sh_main' 和 'sh_star')
                    for m_info in SUPPORTED_MARKETS.values():
                        if m_info.get('code') == market:
                            prefixes.extend(m_info.get('prefixes', []))
            
            if prefixes:
                df = df[df['code'].str.startswith(tuple(prefixes))]
        
        return df


# 单个任务超时阈值（秒）。单只股票超过此时间未完成则跳过，防止卡死
try:
    from config import TASK_TIMEOUT_SECONDS
except ImportError:
    TASK_TIMEOUT_SECONDS = 120

def _kill_executor_workers(executor):
    """强杀进程池里的工作进程。

    进程池里的任务一旦真正卡死（baostock 服务端不响应），``future.cancel()`` 对
    已在运行的任务无效 —— 工作进程会永久占用槽位，后续排队任务永远得不到执行。
    唯一能释放槽位的手段就是杀掉进程；池随之进入 BrokenProcessPool，
    剩余任务交回调用方在新池里续跑。
    """
    try:
        for proc in list(getattr(executor, '_processes', {}).values()):
            try:
                proc.kill()
            except Exception:
                pass
    except Exception:
        pass


def _drain_futures(futures: dict, pbar, task_label: str = "", executor=None):
    """
    带超时的 future 结果收集器。

    返回 ``(remaining_codes, killed, abort_reason)``：
      - ``remaining_codes``: 未被成功处理的股票代码（交回调用方续跑）；
      - ``killed``: 是否因为「单只卡死」而强杀了工作进程（True 时可以安全续跑）；
      - ``abort_reason``: 整批中止的原因，``'quota'`` / ``'systemic'`` / ``None``。
        非 None 时**不要**重建进程池续跑 —— 重试不会成功，只会再烧一遍配额。

    ⚠ 旧实现的行为是错的：任意一只股票卡住超过 ``TASK_TIMEOUT_SECONDS``，就走进
    ``TimeoutError`` 分支把**全部**剩余 future 一次性 cancel —— 实测 2026-09-23
    只更新了 5212 只里的 1037 只（列表前 1173 只），剩下 4376 只从未被尝试，
    而任务却"成功"结束了，日志上完全看不出来。

    新语义：
      1. 正常完成 → 记账 + 推进进度条；
      2. 单只运行超过 ``TASK_HARD_TIMEOUT_SECONDS`` → 只放弃这一只，并强杀占用它的
         工作进程（否则唯一槽位被永久占用），其余任务继续；
      3. 连续 ``TASK_MAX_STALLS`` 轮「没有任何任务完成、也没有任务在运行」→
         判定系统性故障（登录失效 / 网络断），才放弃整批；
      4. 今日配额打满 → 立即停批，剩余部分靠下次增量续跑（DB 已是 checkpoint）。
    """
    from concurrent.futures import wait, FIRST_COMPLETED

    pending = set(futures.keys())
    # 提交顺序：ProcessPoolExecutor 一旦把任务放进调用队列就会把 future 置成 RUNNING，
    # 所以「排队中」和「正在执行」无法用 future.running() 区分。工作进程是 FIFO 消费，
    # 因此**提交顺序最靠前的未完成 future** 才是真正在执行的那一个 —— 计时只给它记。
    order = list(futures.keys())
    n_exec = 1
    if executor is not None:
        try:
            n_exec = max(1, int(getattr(executor, '_max_workers', 1) or 1))
        except Exception:
            n_exec = 1
    run_since = {}
    killed = False
    last_progress = time.time()
    last_quota_check = time.time()
    # 「系统性故障」的判定门槛必须**高于**单只硬超时，否则整批放弃会先于单只放弃触发，
    # 又退化成老行为。零进展时长超过这个门槛（且队首任务也没到硬超时）才放弃整批。
    systemic_deadline = TASK_HARD_TIMEOUT_SECONDS + TASK_MAX_STALLS * TASK_TIMEOUT_SECONDS

    def _settle(future):
        code = futures[future]
        try:
            future.result(timeout=0)
        except Exception as e:
            print(f"\n进程执行失败 [{task_label}] {code}: {e}")
        finally:
            pbar.update(1)
            pending.discard(future)
            run_since.pop(future, None)

    while pending:
        done, _not_done = wait(list(pending), timeout=TASK_TIMEOUT_SECONDS,
                               return_when=FIRST_COMPLETED)
        if done:
            last_progress = time.time()
            for future in done:
                _settle(future)
            # 额度打满后每个 worker 都会在 _register_api_call 里立刻抛异常，
            # future 会以每秒几百个的速度「完成」，但一个数据也拿不到。不在这里
            # 掐掉的话，剩下的几千只每只还要开 8 次 SQLite 连接、打 8 行错误
            # —— 2026-10-05~07 三晚白跑几小时就是这么来的。每 5s 查一次额度，
            # 开销可忽略。返回的 remaining 不带进下一轮重建，交给下次增量续跑。
            if time.time() - last_quota_check >= 5.0:
                last_quota_check = time.time()
                if quota_remaining() == 0:
                    # 先把未处理的记下来再 discard —— 下面那个 cancel 循环会把
                    # pending 清空，事后再算 remaining 只会得到空列表，调用方
                    # 就报不出「还剩多少只」了。
                    unprocessed = [futures[f] for f in order if f in pending]
                    print(f"\n✗ [{task_label}] 今日 Baostock 配额已耗尽，立即停止本阶段；"
                          f"剩余 {len(unprocessed)} 只未处理（已落库的部分就是 checkpoint，"
                          f"下次增量不会重做）")
                    for future in list(pending):
                        future.cancel()
                        pending.discard(future)
                        pbar.update(1)
                    return unprocessed, False, 'quota'
            continue

        # 一轮超时且零进展：先看是不是有任务真的卡死了。
        # 只给「队首的 n_exec 个」计时——它们才是工作进程真正在执行的任务，
        # 其余排在后面的只是被提前 dispatched 到调用队列，不该算它们超时。
        now = time.time()
        heads = [f for f in order if f in pending][:n_exec]
        stuck = []
        for future in heads:
            run_since.setdefault(future, now)
            if now - run_since[future] >= TASK_HARD_TIMEOUT_SECONDS:
                stuck.append(future)

        if stuck:
            for future in stuck:
                code = futures[future]
                print(f"\n⚠ [{task_label}] {code} 运行超过 {TASK_HARD_TIMEOUT_SECONDS}s 未返回，"
                      f"判定卡死：终止工作进程，放弃该只（不再重试），其余任务继续")
                future.cancel()
                # 从 pending 里摘掉，避免它在新进程池里被重新提交、再次卡死
                pbar.update(1)
                pending.discard(future)
                run_since.pop(future, None)
            _kill_executor_workers(executor)
            killed = True
            last_progress = time.time()
            break

        idle = time.time() - last_progress
        if idle >= systemic_deadline:
            unprocessed = [futures[f] for f in order if f in pending]
            print(f"\n✗ [{task_label}] 连续 {idle:.0f}s 既无任务完成也无任务卡死判定，"
                  f"判定系统性故障（登录失效/网络断），放弃剩余 {len(unprocessed)} 只")
            for future in list(pending):
                future.cancel()
                pending.discard(future)
                pbar.update(1)
            return unprocessed, killed, 'systemic'
        print(f"\n⏳ [{task_label}] 已 {idle:.0f}s 无任务完成（上限 {systemic_deadline:.0f}s），继续等待...")

    # 按提交顺序返回，保证续跑顺序稳定
    remaining = [futures[f] for f in futures if f in pending]
    return remaining, killed, None


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
    import socket
    pid = os.getpid()

    # 子进程内设置 socket 默认超时：baostock 的 login()/recv() 没有超时参数，
    # 服务端不响应时会永久阻塞（曾让整个定时任务挂起 12 天）。子进程是独立解释器，
    # 在这里设置不会影响主进程的其他网络库。
    socket.setdefaulttimeout(BAOSTOCK_SOCKET_TIMEOUT)

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
