# 股票筛选模块
import pandas as pd
import sqlite3
from datetime import datetime
from core.data import DataFetcher
from core.indicators import TechnicalIndicators
import os
from config import SELECTION_CRITERIA, DATABASE_PATH

class StockScreener:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.tech_indicators = TechnicalIndicators()
        self.criteria = SELECTION_CRITERIA
    
    def get_candidate_stocks(self):
        """获取候选股票列表（基于基本面筛选:市值、市盈率）"""
        # 确保使用项目根目录的数据库路径
        db_path = DATABASE_PATH
        conn = sqlite3.connect(db_path)

        query = '''
            SELECT DISTINCT s.code, s.name, s.market_cap, s.pe_ratio, s.pb_ratio
            FROM stock_info s
            INNER JOIN daily_data d ON s.code = d.code
            WHERE (
                s.market_cap >= ?
                AND s.pe_ratio <= ?
            )
            GROUP BY s.code
            HAVING COUNT(d.date) >= 30
        '''
        
        candidates = pd.read_sql_query(
            query, conn, 
            params=(self.criteria['min_market_cap'], self.criteria['max_pe'])
        )
        
        conn.close()
        return candidates
    
    def apply_technical_filters(self, stock_code):
        """应用技术指标筛选"""
        # 获取股票数据
        stock_data = self.data_fetcher.get_stock_data(stock_code)
        

        
        # 价格筛选
        latest_price = stock_data['close'].iloc[-1]
        if latest_price < self.criteria['min_price'] or latest_price > self.criteria['max_price']:
            return False, f"价格不符合条件: {latest_price}"
        
        # # 换手率筛选
        # if 'turnover_rate' in stock_data.columns:
        #     avg_turnover_rate = stock_data['turnover_rate'].tail(5).mean()
        #     if pd.isna(avg_turnover_rate) or avg_turnover_rate < self.criteria['min_turnover_rate']:
        #         return False, f"换手率不足: {avg_turnover_rate if not pd.isna(avg_turnover_rate) else 'N/A'}"
        else:
            # 如果没有换手率字段，使用成交量作为备选
            avg_volume = stock_data['volume'].tail(5).mean()
            min_volume = 50000  # 备选最小成交量
            if avg_volume < min_volume:
                return False, f"成交量不足: {avg_volume}"

        
        return True, "通过基础筛选"
    

    
    def close(self):
        """关闭资源"""
        self.data_fetcher.close()