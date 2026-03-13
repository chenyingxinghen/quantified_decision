
import sqlite3
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict

# 确保能找到项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import DATABASE_PATH

class MarketSentimentCalculator:
    """
    全市场情绪指标计算器
    汇总全市场 A 股的运行状态，包括涨跌家数比、破位/突破比例、成交量比例等
    信息存储在 stock_meta.db 的 market_sentiment 表中。
    """
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        self.daily_db = os.path.join(db_dir, 'stock_daily.db')
        self.meta_db = os.path.join(db_dir, 'stock_meta.db')
        
        # 针对全量计算中的窗口函数可能会占用极大临时磁盘空间（C盘）并导致 "database or disk is full" 错误，
        # 我们将 SQLite 的默认临时目录切换到数据库所在的磁盘。
        import tempfile
        temp_dir = os.path.join(db_dir, "system_data", "temp_sqlite")
        os.makedirs(temp_dir, exist_ok=True)
        os.environ['TMP'] = temp_dir
        os.environ['TEMP'] = temp_dir
        tempfile.tempdir = temp_dir

    def check_and_update(self, force: bool = False):
        """
        检查并更新市场情绪数据。
        
        参数:
            force: 是否强制重新完整计算（默认只增量更新缺失日期）
        """
        if not os.path.exists(self.daily_db):
            print(f"警告: 找不到日线数据库 {self.daily_db}")
            return
            
        print("正在检查市场情绪数据更新状态...")
        
        conn_daily = sqlite3.connect(self.daily_db)
        
        # 1. 获取 daily_data 中的所有日期
        dates_df = pd.read_sql_query("SELECT DISTINCT date FROM daily_data ORDER BY date ASC", conn_daily)
        all_daily_dates = set(dates_df['date'].tolist())
        
        # 2. 获取已计算的日期
        computed_dates = set()
        if not force and os.path.exists(self.meta_db):
            conn_meta = sqlite3.connect(self.meta_db)
            try:
                cursor = conn_meta.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_sentiment'")
                if cursor.fetchone():
                    comp_df = pd.read_sql_query("SELECT date FROM market_sentiment", conn_meta)
                    computed_dates = set(comp_df['date'].tolist())
            except Exception as e:
                print(f"检查已计算日期失败: {e}")
            finally:
                conn_meta.close()
        
        # 3. 确定需要更新的日期
        missing_dates = sorted(list(all_daily_dates - computed_dates))
        
        if not missing_dates:
            print("市场情绪数据已是最新，无需更新。")
            conn_daily.close()
            return

        print(f"发现 {len(missing_dates)} 个日期需要计算情绪指标...")
        
        # 为了计算 MA20 广度，我们需要连续的数据，所以如果是增量更新且日期较少，
        # 我们仍需要回溯一部分数据来计算广度，或者为了简化逻辑，如果缺失日期超过一定数量，直接全量重算
        
        if len(missing_dates) > 100 or force:
            self._full_calculate(conn_daily)
        else:
            self._incremental_update(conn_daily, sorted(list(all_daily_dates)), missing_dates)
            
        conn_daily.close()

    def _get_calculation_df(self, conn_daily: sqlite3.Connection, target_dates: List[str], all_dates: List[str]):
        """
        核心计算逻辑：计算指定目标日期的市场情绪指标
        为了 MA20 和上一日收盘价，需要包含足够的回溯数据
        """
        if not target_dates:
            return pd.DataFrame()
            
        min_target = min(target_dates)
        # 获取回溯开始日期 (60个交易日足够覆盖 MA20)
        try:
            idx = all_dates.index(min_target)
            start_idx = max(0, idx - 60)
            data_start_date = all_dates[start_idx]
        except:
            data_start_date = min_target

        print(f"  正在读取行情数据 (从 {data_start_date} 起)...")
        
        # A. 基础统计 (带上一日收盘价以计算涨跌幅)
        # 使用参数化查询过滤日期
        placeholders = ','.join(['?'] * len(target_dates))
        query = f"""
        SELECT 
            date,
            COUNT(*) as total_count,
            SUM(CASE WHEN close > open * 1.0001 THEN 1 ELSE 0 END) as up_count,
            SUM(CASE WHEN (close/open - 1) > 0.05 THEN 1 ELSE 0 END) as strong_up_count,
            SUM(CASE WHEN close < open * 0.9999 THEN 1 ELSE 0 END) as down_count,
            SUM(CASE WHEN close >= high * 0.9999 AND (close/prev_close - 1) > 0.093 THEN 1 ELSE 0 END) as limit_up_count,
            SUM(CASE WHEN close <= low * 1.0001 AND (close/prev_close - 1) < -0.093 THEN 1 ELSE 0 END) as limit_down_count,
            AVG((close/open) - 1) as avg_ret,
            SUM(volume) as total_vol,
            SUM(CASE WHEN close > open THEN volume ELSE 0 END) as adv_vol
        FROM (
            SELECT *, LAG(close) OVER (PARTITION BY code ORDER BY date) as prev_close
            FROM daily_data
            WHERE date >= ?
        )
        WHERE date IN ({placeholders})
        GROUP BY date
        """
        
        params = [data_start_date] + target_dates
        try:
            basic_stats = pd.read_sql_query(query, conn_daily, params=params)
        except Exception as e:
            # 兼容低版本或异常情况
            print(f"  警告: 窗口函数 SQL 失败 ({e})，执行降级汇总...")
            query_fallback = f"""
            SELECT date, COUNT(*) as total_count, 
                   SUM(CASE WHEN close > open THEN 1 ELSE 0 END) as up_count,
                   SUM(CASE WHEN (close/open - 1) > 0.05 THEN 1 ELSE 0 END) as strong_up_count,
                   SUM(CASE WHEN close < open THEN 1 ELSE 0 END) as down_count,
                   AVG((close/open) - 1) as avg_ret,
                   SUM(volume) as total_vol,
                   SUM(CASE WHEN close > open THEN volume ELSE 0 END) as adv_vol
            FROM daily_data WHERE date IN ({placeholders})
            GROUP BY date
            """
            basic_stats = pd.read_sql_query(query_fallback, conn_daily, params=target_dates)
            basic_stats['limit_up_count'] = 0
            basic_stats['limit_down_count'] = 0

        # B. 市场广度 (MA20)
        print("  正在计算 MA20 广度...")
        prices = pd.read_sql_query(
            "SELECT code, date, close FROM daily_data WHERE date >= ? ORDER BY code, date", 
            conn_daily, params=[data_start_date]
        )
        if not prices.empty:
            prices['ma20'] = prices.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
            prices['above_ma20'] = (prices['close'] > prices['ma20']).fillna(False).astype(int)
            
            # 只保留目标日期的广度
            breadth_target = prices[prices['date'].isin(target_dates)]
            breadth_stats = breadth_target.groupby('date')['above_ma20'].mean().reset_index()
            breadth_stats.columns = ['date', 'breadth_ma20']
        else:
            breadth_stats = pd.DataFrame(columns=['date', 'breadth_ma20'])

        # C. 合并
        basic_stats['up_ratio'] = basic_stats['up_count'] / basic_stats['total_count'].replace(0, 1)
        basic_stats['strong_up_ratio'] = basic_stats['strong_up_count'] / basic_stats['total_count'].replace(0, 1)
        basic_stats['down_ratio'] = basic_stats['down_count'] / basic_stats['total_count'].replace(0, 1)
        basic_stats['limit_up_ratio'] = basic_stats.get('limit_up_count', 0) / basic_stats['total_count'].replace(0, 1)
        basic_stats['limit_down_ratio'] = basic_stats.get('limit_down_count', 0) / basic_stats['total_count'].replace(0, 1)
        basic_stats['adv_vol_ratio'] = basic_stats['adv_vol'] / basic_stats['total_vol'].replace(0, 1)
        basic_stats['mean_return'] = basic_stats['avg_ret']
        basic_stats['total_volume'] = basic_stats['total_vol']

        final_df = pd.merge(basic_stats, breadth_stats, on='date', how='left')
        return final_df

    def _full_calculate(self, conn_daily: sqlite3.Connection):
        """全量计算逻辑"""
        print("执行全量情绪指标计算...")
        # 获取所有日期
        all_dates = pd.read_sql_query("SELECT DISTINCT date FROM daily_data ORDER BY date ASC", conn_daily)['date'].tolist()
        final_df = self._get_calculation_df(conn_daily, all_dates, all_dates)
        self._save_to_db(final_df, mode='replace')

    def _incremental_update(self, conn_daily: sqlite3.Connection, all_dates: List[str], missing_dates: List[str]):
        """增量更新逻辑"""
        print(f"执行增量情绪指标计算 ({len(missing_dates)} 个日期)...")
        final_df = self._get_calculation_df(conn_daily, missing_dates, all_dates)
        self._save_to_db(final_df, mode='append')

    def _save_to_db(self, df: pd.DataFrame, mode: str = 'replace'):
        if df.empty: return
        
        save_cols = [
            'date', 'up_ratio', 'strong_up_ratio', 'down_ratio', 
            'limit_up_ratio', 'limit_down_ratio', 'mean_return', 
            'total_volume', 'adv_vol_ratio', 'breadth_ma20'
        ]
        
        conn_meta = sqlite3.connect(self.meta_db)
        try:
            if mode == 'replace':
                df[save_cols].to_sql('market_sentiment', conn_meta, if_exists='replace', index=False)
            else:
                # 增量模式：先删除已存在的日期（防止重复），再追加
                target_dates = df['date'].tolist()
                placeholders = ','.join(['?'] * len(target_dates))
                cursor = conn_meta.cursor()
                # 检查表是否存在
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_sentiment'")
                if cursor.fetchone():
                    cursor.execute(f"DELETE FROM market_sentiment WHERE date IN ({placeholders})", target_dates)
                
                df[save_cols].to_sql('market_sentiment', conn_meta, if_exists='append', index=False)
            
            cursor = conn_meta.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_date ON market_sentiment(date)")
            conn_meta.commit()
            print(f"成功保存 {len(df)} 条市场情绪记录。")
        finally:
            conn_meta.close()

if __name__ == "__main__":
    calc = MarketSentimentCalculator()
    calc.check_and_update()
