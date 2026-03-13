import pandas as pd
import sqlite3
import os
import sys

import requests
import time
import json
from datetime import datetime, timedelta
from urllib.parse import quote
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from config import DATABASE_PATH, YEARS, WORKERS_NUM
from typing import List, Dict, Optional, Tuple

class DataFetcher:
    """
    Unified DataFetcher
    - Yahoo Finance: Daily K-line (OHLCV)
    - EastMoney: Turnover Rate & Financial Reports
    Updates and queries multiple attached SQLite databases.
    """
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.db_dir = os.path.dirname(self.db_path)
        self._conn = None
        self.init_database()

    @property
    def conn(self):
        if self._conn is None:
            self._conn = self._get_conn()
        return self._conn

    def _get_conn(self):
        """Get connection with attached databases (meta, finance)"""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL')
        
        # Attach other databases if they exist
        db_files = {
            'meta': 'stock_meta.db',
            'finance': 'stock_finance.db'
        }
        for alias, filename in db_files.items():
            path = os.path.join(self.db_dir, filename)
            if os.path.exists(path):
                conn.execute(f"ATTACH DATABASE '{path}' AS {alias}")
        return conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def init_database(self):
        """Ensure core tables exist in the main database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_data (
                    code TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    turnover_rate REAL,
                    PRIMARY KEY (code, date)
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_data (date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_code_date ON daily_data (code, date)')

    def _to_yf_symbol(self, code: str) -> str:
        """Convert A-stock code to yfinance symbol"""
        if code.startswith(('60', '68')): return f"{code}.SS"
        if code.startswith(('00', '30')): return f"{code}.SZ"
        if code.startswith(('43', '83', '87', '92')): return f"{code}.BJ"
        return code

    def _to_em_secid(self, code: str) -> str:
        """Convert A-stock code to EastMoney secid"""
        market = 1 if code.startswith('6') else 0
        return f"{market}.{code}"

    def update_daily_data(self, symbol: str, incremental: bool = True):
        """
        Updates daily data for a stock using:
        - Yahoo (OHLCV)
        - EastMoney (Turnover)
        - EastMoney (Finance)
        """
        try:
            # 1. Determine date range
            end_date = datetime.now()+timedelta(days=1)
            if incremental:
                last_update = self._get_last_update_date(symbol)
                if last_update:
                    start_date = datetime.strptime(last_update, '%Y-%m-%d')
                else:
                    print(f"{symbol} no last records in database")
                    start_date = datetime.now() - timedelta(days=365 * config.YEARS)
            else:
                start_date = datetime.now() - timedelta(days=365 * config.YEARS)

            start_str = (start_date+timedelta(days=1)).strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            # 数据库最新日：t，获取数据t+1到now+1
            if datetime.now().time()<datetime.strptime("15:30", "%H:%M").time() and datetime.now().day-start_date.day==1:
                if start_date.strftime('%Y-%m-%d')==(datetime.now()-timedelta(days=1)).strftime('%Y-%m-%d'):
                    return
            elif datetime.now().time()>datetime.strptime("15:30", "%H:%M").time() and datetime.now().day-start_date.day==0:
                return
            

            # 2. Yahoo Request (OHLCV)
            yf_symbol = self._to_yf_symbol(symbol)
            # For yfinance 1.2.0+, download might return MultiIndex
            ohlcv = yf.download(yf_symbol, start=start_str, end=end_str, repair=True, auto_adjust=True,progress=False)

            if ohlcv.empty:
                print(f'{start_date}-{end_date}')
                print(f"{symbol} no records in yfinance")
                return
            # Flatten MultiIndex if ticker is in columns
            if isinstance(ohlcv.columns, pd.MultiIndex):
                # If single ticker, it usually has (Attribute, Ticker) structure
                if yf_symbol in ohlcv.columns.get_level_values(1):
                    ohlcv = ohlcv.xs(yf_symbol, axis=1, level=1)
                else:
                    ohlcv.columns = ohlcv.columns.get_level_values(0)
            
            ohlcv = ohlcv.reset_index()
            
            # Map standard columns
            col_map = {
                'Date': 'date', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }
            ohlcv = ohlcv.rename(columns=col_map)
            
            # Ensure 'date' is string
            if 'date' in ohlcv.columns:
                ohlcv['date'] = pd.to_datetime(ohlcv['date']).dt.strftime('%Y-%m-%d')
            else:
                # Some versions might have 'index' as date
                if 'index' in ohlcv.columns:
                    ohlcv = ohlcv.rename(columns={'index': 'date'})
                    ohlcv['date'] = pd.to_datetime(ohlcv['date']).dt.strftime('%Y-%m-%d')
            
            ohlcv['code'] = symbol

            # 3. EastMoney Request (Turnover Rate,amount)
            secid = self._to_em_secid(symbol)
            beg = start_str.replace('-', '')
            end_em = end_str.replace('-', '')
            em_url = (
                f"https://push2his.eastmoney.com/api/qt/stock/kline/get?"
                f"secid={secid}&klt=101&fqt=1&beg={beg}&end={end_em}&"
                f"fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f56,f61"
            )
            em_resp = requests.get(em_url, timeout=10).json()
            klines = em_resp.get('data', {}).get('klines', []) if em_resp.get('data') else []
            
            em_data_map = {}
            for k in klines:
                if ',' in k:
                    parts = k.split(',')
                    if len(parts) >= 3:
                        d, amt_val, tr_val = parts[0], parts[1], parts[2]
                        em_data_map[d] = (float(amt_val), float(tr_val))

            # 4. Merge and Insert
            cursor = self.conn.cursor()
            for _, row in ohlcv.iterrows():
                try:
                    d = str(row['date'])
                    amt, tr = em_data_map.get(d, (0.0, 0.0))
                    if amt == 0.0:
                        amt = float(row.get('volume', 0)) * float(row.get('close', 0))
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_data 
                        (code, date, open, high, low, close, volume, amount, turnover_rate)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, d, float(row['open']), float(row['high']), float(row['low']),
                          float(row['close']), float(row['volume']), float(amt), float(tr)))
                except Exception as e:
                    import traceback
                    print(f'Error update daily_data: {symbol}: {e}')
                    traceback.print_exc()
            
            self.conn.commit()
            
            # 5. EastMoney Finance Request

            self._update_finance_data(symbol)

        except Exception as e:
            import traceback
            print(f"Error updating {symbol}: {e}")
            traceback.print_exc()

    def _is_report_season(self) -> bool:
        """
        判断当前是否为财报发布高峰期
        
        A 股财报发布时间规律：
        - 年报：次年 1-4 月
        - 一季报：当年 4 月
        - 中报：当年 7-8 月
        - 三季报：当年 10 月
        
        Returns:
            bool: True 如果是财报发布高峰期，False 否则
        """
        current_month = datetime.now().month
        report_months = [1, 2, 3, 4, 8, 10]
        return current_month in report_months
    
    def _update_finance_data(self, symbol: str):
        """Fetch and update financial reports from EastMoney"""
        try:
            # 非财报发布高峰期，跳过财务数据请求以节省资源和避免错误
            if not self._is_report_season():
                return
            
            market_suffix = ".SH" if symbol.startswith('6') else ".SZ"
            secucode = f"{symbol}{market_suffix}"
            
            # Use a conservative date to check for new reports
            date_min = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            type_name = "RPT_F10_FINANCE_MAINFINADATA"
            sty = "APP_F10_MAINFINADATA"
            filter_str = f'(SECUCODE="{secucode}")(REPORT_DATE >= \'{date_min}\')'
            filter_encoded = quote(filter_str, safe='()=><\'"')
            
            url = (
                "https://datacenter.eastmoney.com/securities/api/data/get?"
                f"type={type_name}&sty={sty}&filter={filter_encoded}"
                "&p=1&ps=10&sr=-1&st=REPORT_DATE&source=HSF10&client=PC"
            )
            
            resp = requests.get(url, timeout=10).json()
            if not resp.get("success") or not resp.get("result"):
                return
            
            reports = resp["result"].get("data", [])
            if not reports:
                return

            # Insert into stock_finance.db via attached alias
            cursor = self.conn.cursor()
            
            # Get valid columns for the finance_reports table to avoid schema errors
            cursor.execute("PRAGMA finance.table_info(finance_reports)")
            valid_cols = {info[1] for info in cursor.fetchall()}
            
            # We assume finance_reports table exists in attached 'finance' database
            # Column names from reports dict keys
            for r in reports:
                # Only keep columns that are present in the table
                filtered_r = {k: v for k, v in r.items() if k in valid_cols}
                
                # Add 'code' column if it's in valid_cols and not present
                if 'code' in valid_cols and 'code' not in filtered_r:
                    filtered_r['code'] = symbol

                # 统一 ORG_TYPE 为数字编码
                if 'ORG_TYPE' in filtered_r:
                    ot = filtered_r['ORG_TYPE']
                    if ot == "银行": ot = 1
                    elif ot == "证券": ot = 2
                    elif ot == "保险": ot = 3
                    elif isinstance(ot, str) and not ot.isdigit(): ot = 0
                    filtered_r['ORG_TYPE'] = int(ot) if ot is not None else 0
                
                # 统一报告日期格式 (截断时间部分 YYYY-MM-DD)
                if 'REPORT_DATE' in filtered_r and filtered_r['REPORT_DATE']:
                    filtered_r['REPORT_DATE'] = str(filtered_r['REPORT_DATE'])[:10]
                
                # 统一公告日期格式
                if 'NOTICE_DATE' in filtered_r and filtered_r['NOTICE_DATE']:
                    filtered_r['NOTICE_DATE'] = str(filtered_r['NOTICE_DATE'])[:10]
                
                if not filtered_r:
                    continue
                    
                cols = list(filtered_r.keys())
                
                placeholders = ', '.join(['?'] * len(cols))
                col_names = ', '.join(cols)
                vals = [filtered_r[c] for c in cols]
                
                # Use finance.finance_reports
                cursor.execute(f'''
                    INSERT OR REPLACE INTO finance.finance_reports ({col_names})
                    VALUES ({placeholders})
                ''', vals)
            self.conn.commit()
        except Exception as e:
            print(f"Error updating finance data for {symbol}: {e}")

    def _get_last_update_date(self, symbol: str) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(date) FROM daily_data WHERE code = ?", (symbol,))
        res = cursor.fetchone()
        return res[0] if res and res[0] else None

    # Utility Methods (Maintaining original interface for select_stocks/data_handler)
    
    def get_stock_list(self, markets=config.DEFAULT_MARKETS) -> pd.DataFrame:
        """Get stock list from meta.stock_info_extended"""
        try:
            query = "SELECT code, name FROM meta.stock_info_extended"
            df = pd.read_sql_query(query, self.conn)
            
            pref_map = {'sh': '60', 'sz_main': '00', 'sz_gem': '30', 'bj': ('8', '4')}
            prefixes = []
            for m in markets:
                p = pref_map.get(m)
                if isinstance(p, tuple): prefixes.extend(p)
                else: prefixes.append(p)
            
            if prefixes:
                df = df[df['code'].str.startswith(tuple(prefixes))]
            return df
        except Exception:
            return pd.DataFrame()

    def get_stock_data(self, symbol: str, days: int = 1) -> pd.DataFrame:
        """Retrieves last N days of data from database"""
        query = "SELECT * FROM daily_data WHERE code = ? ORDER BY date DESC LIMIT ?"
        df = pd.read_sql_query(query, self.conn, params=(symbol, days))
        return df.sort_values('date').reset_index(drop=True)

    def get_historical_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Historical data for specific date range (Standardized output)"""
        query = "SELECT * FROM daily_data WHERE code = ?"
        params = [symbol]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date ASC"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        # Renaissance mapping for old code consistency
        df = df.rename(columns={
            'date': '日期', 'open': '开盘', 'high': '最高', 
            'low': '最低', 'close': '收盘', 'volume': '成交量',
            'amount': '成交额', 'turnover_rate': '换手率'
        })
        return df

    def get_stock_info_extended(self, code: str = None) -> pd.DataFrame:
        query = "SELECT * FROM meta.stock_info_extended"
        if code:
            return pd.read_sql_query(query + " WHERE code = ?", self.conn, params=(code,))
        return pd.read_sql_query(query, self.conn)

    def get_realtime_data(self, symbol: str) -> pd.DataFrame:
        """Live quote from Yahoo + EastMoney"""
        yf_symbol = self._to_yf_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)
        info = ticker.fast_info
        
        res = {
            '代码': symbol,
            '最新价': info.last_price,
            '成交量': info.last_volume,
            '开盘': info.open,
            '最高': info.day_high,
            '最低': info.day_low,
        }
        return pd.DataFrame([res])

    def init_all_stocks_data(self, markets=['sh', 'sz_main'], incremental: bool = True, workers: int = WORKERS_NUM):
        """Batch update for all stocks"""
        stocks = self.get_stock_list(markets)
        if stocks.empty: return
        
        codes = stocks['code'].tolist()
        print(f"Starting update for {len(codes)} stocks using {workers} workers...")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            def worker(c_symbol):
                fetcher = DataFetcher()
                try:
                    fetcher.update_daily_data(c_symbol, incremental)
                finally:
                    fetcher.close()

            fut_map = {executor.submit(worker, c): c for c in codes}
            for fut in as_completed(fut_map):
                c = fut_map[fut]
                try:
                    fut.result()
                except Exception as e:
                    import traceback
                    print(f"Failed {c}: {e}")
                    traceback.print_exc()