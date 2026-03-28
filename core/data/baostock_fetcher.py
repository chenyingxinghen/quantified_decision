"""
Baostock 数据获取器 - 基础类
核心特性：
1. 使用不复权数据 + 前复权因子实现动态前复权
2. 获取完整的财务基本面数据
3. 统一使用 daily_data 作为主表名
"""
import pandas as pd
import sqlite3
import os
import sys
import baostock as bs
from datetime import datetime
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.baostock_config import DATABASE_PATH, META_DB_PATH, FINANCE_DB_PATH

# Baostock API 内部不再使用全局锁


class BaostockFetcher:
    """
    Baostock 数据获取器基类
    """
    _global_bs_logged_in = False
    
    _db_initialized = False
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.db_dir = os.path.dirname(self.db_path)
        self._conn = None
        if not BaostockFetcher._db_initialized:
            self.init_database()
            BaostockFetcher._db_initialized = True
    
    @property
    def conn(self):
        if self._conn is None:
            self._conn = self._get_conn()
        return self._conn
    
    def _get_conn(self):
        """获取数据库连接并附加其他数据库"""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        # WAL 模式和 Synchronous 设置移动到 init_database 中，不再每次连接都设置，减少写事务开销
        conn.row_factory = sqlite3.Row
        
        # 附加其他数据库
        db_configs = {
            'meta': META_DB_PATH,
            'finance': FINANCE_DB_PATH
        }
        for alias, path in db_configs.items():
            if os.path.exists(path):
                conn.execute(f"ATTACH DATABASE '{path}' AS {alias}")
        return conn

    def safe_commit(self, max_retries: int = 5, initial_delay: float = 0.5):
        """
        带重试机制的提交，解决 sqlite3 "database is locked" 问题
        """
        import time
        import random
        
        last_err = None
        for i in range(max_retries):
            try:
                self.conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    last_err = e
                    # 随机指数避让
                    delay = initial_delay * (2 ** i) + random.random()
                    time.sleep(delay)
                    continue
                else:
                    raise
            except Exception as e:
                raise e
        
        if last_err:
            print(f"  ⚠ 数据库提交多次重试失败 (PID: {os.getpid()}): {last_err}")
            raise last_err
    
    @classmethod
    def _bs_login(cls) -> bool:
        """登录 baostock"""
        if not cls._global_bs_logged_in:
            lg = bs.login()
            if lg.error_code == '0':
                cls._global_bs_logged_in = True
                print(f"Baostock 登录成功 (PID: {os.getpid()})")
            else:
                print(f"Baostock 登录失败 (PID: {os.getpid()}): {lg.error_msg}")
                # 如果是特定错误，可以不抛异常，由上层处理
                return False
        return cls._global_bs_logged_in
    
    def close(self):
        """关闭数据库连接 (不注销 baostock，由管理器统一注销)"""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def logout(self):
        """显式注销 baostock"""
        if BaostockFetcher._global_bs_logged_in:
            bs.logout()
            BaostockFetcher._global_bs_logged_in = False
            print("Baostock 已注销")
    
    def init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            # 在初始化时设置一次即可，WAL 模式是数据库级别持久的
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=OFF') 
            
            # 统一使用 daily_data 表名
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_data (
                    code TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    preclose REAL,
                    volume REAL,
                    amount REAL,
                    adjustflag TEXT,
                    turnover_rate REAL,
                    tradestatus INTEGER,
                    pctChg REAL,
                    peTTM REAL,
                    pbMRQ REAL,
                    psTTM REAL,
                    pcfNcfTTM REAL,
                    is_st INTEGER,
                    PRIMARY KEY (code, date)
                )
            ''')
            
            # 复权因子表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS adjust_factor (
                    code TEXT,
                    date TEXT,
                    fore_adjust_factor REAL,
                    back_adjust_factor REAL,
                    PRIMARY KEY (code, date)
                )
            ''')
            
            # 创建索引
            conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_data (date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_code_date ON daily_data (code, date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_adjust_date ON adjust_factor (date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_adjust_code_date ON adjust_factor (code, date)')
        
        # 初始化财务数据库
        self._init_finance_database()
        
        # 初始化元数据数据库
        self._init_meta_database()
        
        # 检查并补齐 daily_data 缺少的列
        self._check_and_update_schema()


    def _check_and_update_schema(self):
        """检查 daily_data 结构，补齐缺少的列 (如 PE/PB 等)"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(daily_data)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # 需要确保存在的列
                new_columns = {
                    'preclose': 'REAL',
                    'adjustflag': 'TEXT',
                    'turnover_rate': 'REAL',
                    'tradestatus': 'INTEGER',
                    'pctChg': 'REAL',
                    'peTTM': 'REAL',
                    'pbMRQ': 'REAL',
                    'psTTM': 'REAL',
                    'pcfNcfTTM': 'REAL',
                    'is_st': 'INTEGER'
                }
                
                for col, dtype in new_columns.items():
                    if col not in columns:
                        # 特殊处理 turn -> turnover_rate 的情况
                        if col == 'turnover_rate' and 'turn' in columns:
                             print(f"检测到旧列名 turn，正在重命名为 turnover_rate...")
                             # SQLite 3.25+ supports RENAME COLUMN, but for compatibility we might just add and copy
                             # But usually we can just add and later system will fill.
                             # Here we add the new one.
                             pass
                        
                        print(f"正在为 daily_data 添加缺失列: {col}")
                        conn.execute(f"ALTER TABLE daily_data ADD COLUMN {col} {dtype}")
                
                # 兼容旧列名（如果存在 turn 但没有 turnover_rate，进行一次性补偿）
                if 'turn' in columns and 'turnover_rate' in columns:
                    conn.execute("UPDATE daily_data SET turnover_rate = turn WHERE turnover_rate IS NULL")
                    # 不删除 turn 以保持安全
                # 检查并补齐财务表索引
                from config.baostock_config import FINANCE_TABLES
                finance_db_path = os.path.join(self.db_dir, 'stock_finance.db')
                if os.path.exists(finance_db_path):
                    with sqlite3.connect(finance_db_path) as f_conn:
                        for table in FINANCE_TABLES:
                            f_conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_code_stat ON {table} (code, stat_date)')
        except Exception as e:
            print(f"架构检查失败 (忽略): {e}")

    def _init_finance_database(self):
        """初始化财务数据库表结构 - 仅初始化配置开启的表"""
        from config.baostock_config import FINANCE_TABLES
        finance_db_path = os.path.join(self.db_dir, 'stock_finance.db')
        with sqlite3.connect(finance_db_path, timeout=30.0) as conn:
            # 盈利能力
            if 'profit_ability' in FINANCE_TABLES:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS profit_ability (
                        code TEXT, pub_date TEXT, stat_date TEXT,
                        roeAvg REAL, npMargin REAL, gpMargin REAL, netProfit REAL,
                        epsTTM REAL, MBRevenue REAL, totalShare REAL, liqaShare REAL,
                        PRIMARY KEY (code, pub_date, stat_date)
                    )
                ''')
            # 营运能力
            if 'operation_ability' in FINANCE_TABLES:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS operation_ability (
                        code TEXT, pub_date TEXT, stat_date TEXT,
                        NRTurnRatio REAL, NRTurnDays REAL, INVTurnRatio REAL, INVTurnDays REAL,
                        CATurnRatio REAL, AssetTurnRatio REAL,
                        PRIMARY KEY (code, pub_date, stat_date)
                    )
                ''')
            # 成长能力
            if 'growth_ability' in FINANCE_TABLES:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS growth_ability (
                        code TEXT, pub_date TEXT, stat_date TEXT,
                        YOYEquity REAL, YOYAsset REAL, YOYNI REAL, YOYEPSBasic REAL, YOYPNI REAL,
                        PRIMARY KEY (code, pub_date, stat_date)
                    )
                ''')
            # 偿债能力
            if 'balance_ability' in FINANCE_TABLES:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS balance_ability (
                        code TEXT, pub_date TEXT, stat_date TEXT,
                        currentRatio REAL, quickRatio REAL, cashRatio REAL, YOYLiability REAL,
                        liabilityToAsset REAL, assetToEquity REAL,
                        PRIMARY KEY (code, pub_date, stat_date)
                    )
                ''')
            # 现金流量
            if 'cash_flow' in FINANCE_TABLES:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS cash_flow (
                        code TEXT, pub_date TEXT, stat_date TEXT,
                        CAToAsset REAL, NCAToAsset REAL, tangibleAssetToAsset REAL,
                        ebitToInterest REAL, CFOToOR REAL, CFOToNP REAL, CFOToGr REAL,
                        PRIMARY KEY (code, pub_date, stat_date)
                    )
                ''')
            # 杜邦指数
            if 'dupont' in FINANCE_TABLES:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS dupont (
                        code TEXT, pub_date TEXT, stat_date TEXT,
                        dupontROE REAL, dupontAssetStoEquity REAL, dupontAssetTurn REAL,
                        dupontPnitoni REAL, dupontNitogr REAL, dupontTaxBurden REAL,
                        dupontIntburden REAL, dupontEbittogr REAL,
                        PRIMARY KEY (code, pub_date, stat_date)
                    )
                ''')
            # 业绩预告
            if 'performance_forecast' in FINANCE_TABLES:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_forecast (
                        code TEXT, pub_date TEXT, stat_date TEXT,
                        profitForcastExpPubDate TEXT, profitForcastType TEXT,
                        profitForcastAbstract TEXT, profitForcastChgPctUp REAL, profitForcastChgPctDwn REAL,
                        PRIMARY KEY (code, pub_date, stat_date)
                    )
                ''')
            
            # 为启用的表添加索引
            for table in FINANCE_TABLES:
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_code_stat ON {table} (code, stat_date)')

    def _init_meta_database(self):
        """初始化元数据数据库表结构"""
        meta_db_path = META_DB_PATH
        with sqlite3.connect(meta_db_path, timeout=30.0) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sync_status (
                    code TEXT PRIMARY KEY,
                    last_daily_sync TEXT,
                    last_finance_sync TEXT
                )
            ''')
            # 基础股票列表缓存
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_basic (
                    code TEXT PRIMARY KEY,
                    code_name TEXT,
                    ipoDate TEXT,
                    outDate TEXT,
                    type TEXT,
                    status TEXT,
                    update_time TEXT
                )
            ''')
            # 基础股票行业缓存
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_industry (
                    code TEXT PRIMARY KEY,
                    update_date TEXT,
                    stock_name TEXT,
                    industry TEXT,
                    industry_classification TEXT,
                    update_time TEXT
                )
            ''')

    def _get_sync_status(self, code: str) -> Dict[str, Optional[str]]:
        """获取同步状态"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT last_daily_sync, last_finance_sync FROM meta.sync_status WHERE code = ?", (code,))
            row = cursor.fetchone()
            if row:
                return {'daily': row[0], 'finance': row[1]}
        except:
            pass
        return {'daily': None, 'finance': None}

    def _get_stock_ipo_year(self, code: str) -> Optional[int]:
        """从缓存获取股票上市年份"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT ipoDate FROM meta.stock_basic WHERE code = ?", (code,))
            row = cursor.fetchone()
            if row and row[0]:
                return int(row[0][:4])
        except:
            pass
        return None

    def _get_stock_out_year(self, code: str) -> Optional[int]:
        """从缓存获取股票退市年份"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT outDate FROM meta.stock_basic WHERE code = ?", (code,))
            row = cursor.fetchone()
            if row and row[0]:
                return int(row[0][:4])
        except:
            pass
        return None

    def _update_sync_status(self, code: str, sync_type: str, value: str):
        """更新同步状态"""
        cursor = self.conn.cursor()
        col = 'last_daily_sync' if sync_type == 'daily' else 'last_finance_sync'
        try:
            cursor.execute(f'''
                INSERT INTO meta.sync_status (code, {col}) VALUES (?, ?)
                ON CONFLICT(code) DO UPDATE SET {col} = excluded.{col}
            ''', (code, value))
            self.safe_commit()
        except Exception as e:
            print(f"  ⚠ 更新同步状态失败 ({code}): {e}")

    def _get_stock_basic_from_db(self) -> pd.DataFrame:
        """从元数据库获取股票列表"""
        try:
            # 检查是否有数据
            query = "SELECT code, code_name, ipoDate, outDate, type, status, update_time FROM meta.stock_basic"
            df = pd.read_sql_query(query, self.conn)
            return df
        except:
            return pd.DataFrame()

    def _save_stock_basic_to_db(self, df: pd.DataFrame):
        """将股票列表保存到元数据库"""
        if df.empty: return
        cursor = self.conn.cursor()
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 准备数据，适配 Baostock 原始字段和我们需要的
        rows = []
        for _, row in df.iterrows():
            rows.append((
                row['code'], row.get('code_name', ''), row.get('ipoDate', ''), 
                row.get('outDate', ''), row.get('type', ''), row.get('status', ''),
                now_str
            ))
            
        sql = '''
            INSERT OR REPLACE INTO meta.stock_basic 
            (code, code_name, ipoDate, outDate, type, status, update_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        cursor.executemany(sql, rows)
        self.safe_commit()

    def _get_stock_industry_from_db(self) -> pd.DataFrame:
        """从元数据库获取股票行业信息"""
        try:
            query = "SELECT code, update_date, stock_name, industry, industry_classification FROM meta.stock_industry"
            df = pd.read_sql_query(query, self.conn)
            return df
        except:
            return pd.DataFrame()

    def _save_stock_industry_to_db(self, df: pd.DataFrame):
        """将股票行业信息保存到元数据库"""
        if df.empty: return
        cursor = self.conn.cursor()
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        rows = []
        for _, row in df.iterrows():
            rows.append((
                row['code'], row.get('update_date', ''), row.get('stock_name', ''),
                row.get('industry', ''), row.get('industry_classification', ''),
                now_str
            ))

        sql = '''
            INSERT OR REPLACE INTO meta.stock_industry 
            (code, update_date, stock_name, industry, industry_classification, update_time)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        cursor.executemany(sql, rows)
        self.safe_commit()