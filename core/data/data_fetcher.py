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

    @staticmethod
    def _generate_em_cookie(code: str = None) -> str:
        """
        动态生成东方财富请求所需的 Cookie 字符串。

        各字段生成逻辑：
          qgqp_b_id  : 设备 ID，基于随机 hex。
          st_nvi     : 会话随机标识符 (alnum + 连字符)。
          st_si      : 会话 ID，纯数字 (14 位)。
          st_asi     : 行为链签名，格式同 st_psi（或 'delete' 表示新会话）。
          nid18      : 浏览器指纹，随机 hex。
          nid18_create_time : 毫秒级时间戳。
          gviem      : 访问来源指纹 (alnum + 连字符)。
          gviem_create_time : 毫秒级时间戳。
          fullscreengg / fullscreengg2 : 广告展示状态，固定为 1。
          st_pvi     : 访问次数 (14 位随机)。
          st_sp      : 访问时间 (URL编码的 YYYY-MM-DD%20HH%3AMM%3ASS)。
          st_inirUrl : 来源页面 (URL编码)。
          st_sn      : 小整数计数器 (1–9)。
          st_psi     : 行为链签名，格式: timestamp-ua_code-rand。
        """
        import random
        import string
        import time
        from datetime import datetime
        from urllib.parse import quote

        def rand_hex(n: int) -> str:
            return ''.join(random.choices('0123456789abcdef', k=n))

        def rand_alnum_dash(n: int) -> str:
            """alnum + 少量连字符，模拟东财会话标识符"""
            chars = string.ascii_letters + string.digits
            body = ''.join(random.choices(chars, k=n - 4))
            suffix = ''.join(random.choices(string.digits, k=4))
            return body + suffix

        now_ts_ms = int(time.time() * 1000)
        now_dt    = datetime.now()

        # 1. 设备 ID
        qgqp_b_id = rand_hex(32)

        # 2. 会话标识符
        st_nvi  = rand_alnum_dash(20)
        st_si   = ''.join(random.choices(string.digits, k=14))
        st_pvi  = ''.join(random.choices(string.digits, k=14))
        st_sn   = str(random.randint(1, 9))

        # 3. 浏览器指纹
        nid18   = rand_hex(32)
        gviem   = rand_alnum_dash(15)

        # 时间戳：在当前时间前后随机偏移，模拟真实会话建立时间
        session_offset_ms = random.randint(30_000, 300_000)   # 30s ~ 5min 前起了会话
        create_ts = now_ts_ms - session_offset_ms
        nid18_create_time = str(create_ts)
        gviem_create_time = str(create_ts)

        # 4. 访问时间（URL 编码）
        sp_raw = now_dt.strftime('%Y-%m-%d %H:%M:%S')
        st_sp  = quote(sp_raw, safe='')  # -> '2026-03-16%2023%3A10%3A39'

        # 5. 来源页 URL
        if code:
            mkt = 'sh' if code.startswith('6') else 'sz'
            inir_raw = f'https://quote.eastmoney.com/{mkt}{code}.html'
        else:
            inir_raw = 'https://www.eastmoney.com/'
        st_inirUrl = quote(inir_raw, safe='')

        # 6. 行为链签名 (st_psi / st_asi)
        psi_ts   = now_dt.strftime('%Y%m%d%H%M%S') + ''.join(random.choices(string.digits, k=3))
        ua_code  = '-'.join([
            ''.join(random.choices(string.digits, k=12)),
            ''.join(random.choices(string.digits, k=12)),
            ''.join(random.choices(string.digits, k=10)),
        ])
        st_psi = f"{psi_ts}-{ua_code}"
        st_asi = 'delete'   # 新会话开始时通常为 delete

        parts = [
            f"qgqp_b_id={qgqp_b_id}",
            f"st_nvi={st_nvi}",
            f"st_si={st_si}",
            f"st_asi={st_asi}",
            f"nid18={nid18}",
            f"nid18_create_time={nid18_create_time}",
            f"gviem={gviem}",
            f"gviem_create_time={gviem_create_time}",
            "fullscreengg=1",
            "fullscreengg2=1",
            f"st_pvi={st_pvi}",
            f"st_sp={st_sp}",
            f"st_inirUrl={st_inirUrl}",
            f"st_sn={st_sn}",
            f"st_psi={st_psi}",
        ]
        return '; '.join(parts)

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
                return
            elif datetime.now().time()>datetime.strptime("15:30", "%H:%M").time() and datetime.now().day-start_date.day==0:
                return
            

            # 2. Yahoo Request (OHLCV)
            yf_symbol = self._to_yf_symbol(symbol)
            # For yfinance 1.2.0+, download might return MultiIndex
            _suspended = False  # 停牌标记：yfinance 认为退市但实际是停牌
            try:
                ohlcv = yf.download(yf_symbol, start=start_str, end=end_str,
                                    repair=True, auto_adjust=True, progress=False)
            except Exception as yf_err:
                raise  # 如果有直接抛出的网络层错误，交给外层重试

            if ohlcv.empty:
                # yfinance 对许多错误（包括停牌/查不到数据/网络错误）只打日志、返回空DF，不会抛异常
                # 将真正的错误转为异常以触发外层「指数退避重试」，停牌/空数据则视为 _suspended 继续
                import yfinance.shared as yf_shared
                err_msg = str(yf_shared._ERRORS.get(yf_symbol, ''))
                if err_msg and 'possibly delisted' not in err_msg and 'No data found' not in err_msg:
                    raise Exception(f"YF Err: {err_msg}")  # 强制抛出，触发外层指数退避重试
                
                _suspended = True  # 视为停牌，继续获取 EM 换手率
                ohlcv = pd.DataFrame()

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

            headers = {
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Cookie": 'qgqp_b_id=c91cd0d5d7ffdec45cbd87eaf3be88ab; st_nvi=H9j31DrwvRw7LCTjOKDCO7f23; p_origin=https%3A%2F%2Fpassport2.eastmoney.com; st_si=74321427112891; nid18=02a135560fe3dffacb5c7d217e719939; nid18_create_time=1773683211925; gviem=koQfjPUr4wnclZn_D2L0o88d3; gviem_create_time=1773683211925; mtp=1; ct=oTgrGil11xTUGr6RvFJOCdjSRYyKAZx6drq1PMLdff-pQuZlHREBRe9RPLOLn9F-9k_LXmsBCyCHkPWWhnh3ILBJOd9IyWk3Zw7_PFIRQ7EiTF3cNLPEXRM_ygNh3pNbIunBj3jIC1mbldtDxX-8aKSSY1tHLUcyXKtVhDQXD7g; ut=FobyicMgeV7bfas_M05TDIWd4XHjHsle4qlh_FbvNEAvkBIEwI8RbDW8eEb5qvvJ2d-zRoz9XOdvIXp3MHmt1MxCmH8VK3_xQmdPqaPoD96U-7PD1wp2N-KIKAUZVsrRQYyTvbyFC4XQIMVvlvZrXrv7dtr9DGp1IKed9k0Kc5eIqFQE1LR0LdgRYZACF2yw4TUJb1ASHPU69kuKBmKZfK2_HqMTsTvYbxegEVH1MhoCH9xFShqvQx5S1ocD7skvRU0czL_lSFhlNa7yTNubEm_QGqAKV3dU; pi=4929027679601506%3Bs4929027679601506%3B%E5%B0%98%E5%BD%B1%E6%98%9F%E7%97%95%3BqJgBeMlv%2FmFahZ6EhOxV5tKRUSKP%2FQ%2F8awj9kOI588lokJlu8ZkNrJtvRn%2Fivi9TxLPhbv7a%2Fl7rUJkT2wqtvYxhUwsHlxwWE8JzC7hj2V0Jn%2FLgQ4dRZPNgkcW5HIY8ms7zJaCiCkTM2a7HDcDhuLuQTYLABMK74c6O8dfuD495oRhL3nJ9Y8%2BRpJNr%2Bmv5739ik8Vw%3B4%2F7iTjitgX7cQpQProOey%2FUfSZSxRfb9eLzo8Ylfjwf%2B0nTa3uoHsKd5iIKv4m%2BInM%2F6b1WkRWKDM4Y4a1CIF%2BzAJ9fQ4Avc96TdVrSs8f1zU8U8rsAtpHgdWJymQq%2FB5T5E3LRmo%2Fc6gG5aTIzs9mTWTU2jvg%3D%3D; uidal=4929027679601506%e5%b0%98%e5%bd%b1%e6%98%9f%e7%97%95; sid=; vtpst=|; fullscreengg=1; fullscreengg2=1; wsc_checkuser_ok=1; st_pvi=54188038020678; st_sp=2025-11-03%2022%3A26%3A24; st_inirUrl=https%3A%2F%2Fdata.eastmoney.com%2Fxuangu%2F; st_sn=10; st_psi=20260317090016952-113200354966-7143833211; st_asi=delete',
                "Host": "push2his.eastmoney.com",
                "Pragma": "no-cache",
                'Referer': f'https://quote.eastmoney.com/{"sh" if symbol.startswith("6") else "sz"}{symbol}.html',
                'Sec-Fetch-Site': 'same-site',
                'Sec-Fetch-Dest': 'script',
                'Sec-Fetch-Mode': 'no-cors',
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0",
                "sec-ch-ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Microsoft Edge";v="146"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"'
            }
            if not _suspended:
                em_resp = requests.get(em_url, timeout=10,headers=headers).json()
                klines = em_resp.get('data', {}).get('klines', []) if em_resp.get('data') else []
            
                em_data_map = {}
                for k in klines:
                    if ',' in k:
                        parts = k.split(',')
                        if len(parts) >= 3:
                            d, amt_val, tr_val = parts[0], parts[1], parts[2]
                            em_data_map[d] = (float(amt_val), float(tr_val))
            else:
                em_data_map = {}
            

            # 4. Merge and Insert
            cursor = self.conn.cursor()
            if not ohlcv.empty:
                for _, row in ohlcv.iterrows():
                    try:
                        d = str(row['date'])
                        amt, tr = em_data_map.get(d, (0.0, 0.0))
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO daily_data 
                            (code, date, open, high, low, close, volume, amount, turnover_rate)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol, d, float(row['open']), float(row['high']), float(row['low']),
                              float(row['close']), float(row['volume']), float(amt), float(tr)))
                    except Exception as e:
                        print(f'Error update daily_data: {symbol}: {e}')
            else:
                try:
                    d = start_str
                    last_close = self.get_stock_data(symbol, 1).iloc[0]['close']
                    o, h, l, c, v, a, tr = last_close,last_close,last_close,last_close,0,0,0
                    cursor.execute('''
                        INSERT OR IGNORE INTO daily_data
                        (code, date, open, high, low, close, volume, amount, turnover_rate)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, d, o, h, l, c, v, a, tr))
                except Exception as e:
                    print(f'Error update daily_data: {symbol}: {e}')

            self.conn.commit()
            
            # 5. EastMoney Finance Request

            # self._update_finance_data(symbol)
            time.sleep(config.QUEST_INTERVAL)

        except Exception as e:
            err_msg = str(e)
            if 'possibly delisted' in err_msg or 'No data found' in err_msg:
                # 停牌误报：静默跳过，不打印堆栈
                print(f"Error updating {symbol}: {e}")
            else:
                import traceback
                print(f"Error updating {symbol}: {e}")
                traceback.print_exc()
                raise   # 重新抛出，让上层重试机制捕获

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
        """Batch update for all stocks with exponential-backoff retry"""
        stocks = self.get_stock_list(markets)
        if stocks.empty: return

        codes = stocks['code'].tolist()
        print(f"Starting update for {len(codes)} stocks using {workers} workers...")

        MAX_RETRIES = 3
        BASE_DELAY  = 2.0   # seconds

        def worker(c_symbol):
            """单只股票更新，失败时指数退避重试"""
            for attempt in range(MAX_RETRIES + 1):
                fetcher = DataFetcher()
                try:
                    fetcher.update_daily_data(c_symbol, incremental)
                    return  # 成功，退出重试循环
                except Exception as e:
                    err_msg = str(e)
                    # 停牌误报不重试
                    if 'possibly delisted' in err_msg or 'No data found' in err_msg:
                        return
                    if attempt < MAX_RETRIES:
                        # 指数退避 + 随机扰动: 2^(attempt+1) + rand(0,1) 秒
                        delay = BASE_DELAY ** (attempt + 1) + random.uniform(0, 1)
                        print(f"  [{c_symbol}] 第 {attempt+1} 次失败，{delay:.1f}s 后重试: {e}")
                        time.sleep(delay)
                    else:
                        print(f"  [{c_symbol}] 已达最大重试次数 ({MAX_RETRIES})，放弃: {e}")
                finally:
                    fetcher.close()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            fut_map = {executor.submit(worker, c): c for c in codes}
            for fut in as_completed(fut_map):
                c = fut_map[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"Failed {c}: {e}")