"""
使用yfinance更新stock_info数据库
选取对量化因子构建有用的信息并扩展数据库
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import sqlite3
from datetime import datetime
from config.config import DATABASE_PATH,QUEST_INTERVAL
import time


class StockInfoUpdater:
    """股票信息更新器"""
    
    def __init__(self, db_path=DATABASE_PATH):
        """初始化"""
        self.db_path = db_path
        self.conn = None
        
    def connect_db(self):
        """连接数据库"""
        self.conn = sqlite3.connect(self.db_path)
        
    def close_db(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
    
    def create_extended_table(self):
        """创建扩展的stock_info表"""
        cursor = self.conn.cursor()
        
        # 检查表是否存在
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='stock_info_extended'
        """)
        
        if cursor.fetchone() is None:
            # 创建新表
            cursor.execute('''
                CREATE TABLE stock_info_extended (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    
                    -- 基本信息
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    
                    -- 估值指标
                    pe_ratio REAL,
                    pb_ratio REAL,
                    forward_pe REAL,
                    peg_ratio REAL,
                    price_to_sales REAL,
                    price_to_book REAL,
                    enterprise_to_revenue REAL,
                    enterprise_to_ebitda REAL,
                    
                    -- 财务指标
                    profit_margins REAL,
                    gross_margins REAL,
                    operating_margins REAL,
                    ebitda_margins REAL,
                    return_on_assets REAL,
                    return_on_equity REAL,
                    
                    -- 资产负债
                    total_cash REAL,
                    total_debt REAL,
                    debt_to_equity REAL,
                    current_ratio REAL,
                    quick_ratio REAL,
                    book_value REAL,
                    
                    -- 现金流
                    free_cashflow REAL,
                    operating_cashflow REAL,
                    
                    -- 增长指标
                    revenue_growth REAL,
                    earnings_growth REAL,
                    earnings_quarterly_growth REAL,
                    
                    -- 股息指标
                    dividend_rate REAL,
                    dividend_yield REAL,
                    payout_ratio REAL,
                    five_year_avg_dividend_yield REAL,
                    
                    -- 风险指标
                    beta REAL,
                    
                    -- 股本信息
                    shares_outstanding REAL,
                    float_shares REAL,
                    held_percent_insiders REAL,
                    held_percent_institutions REAL,
                    
                    -- 价格信息
                    current_price REAL,
                    fifty_two_week_high REAL,
                    fifty_two_week_low REAL,
                    fifty_day_average REAL,
                    two_hundred_day_average REAL,
                    
                    -- 成交量
                    average_volume REAL,
                    average_volume_10days REAL,
                    
                    -- 更新时间
                    update_time TIMESTAMP
                )
            ''')
            self.conn.commit()
            print("创建stock_info_extended表成功")
        else:
            print("stock_info_extended表已存在")
    
    def get_stock_list(self):
        from core.data.data_fetcher import DataFetcher
        df=DataFetcher()
        codes=df.get_stock_list()
        symbols=[]
        for code in codes['code']:
            symbols.append(self.convert_stock_code(code))
        return  symbols
    
    def convert_stock_code(self, code):
        """
        将A股代码转换为yfinance格式
        
        参数:
            code: A股代码，如 '000001'
        
        返回:
            yfinance格式代码，如 '000001.SZ'
        """
        if code.startswith('6'):
            return f"{code}.SS"  # 上海
        else:
            return f"{code}.SZ"  # 深圳
    
    def fetch_stock_info(self, code):
        """
        使用yfinance获取股票信息
        
        参数:
            code: A股代码
        
        返回:
            股票信息字典
        """
        yf_code = self.convert_stock_code(code)
        
        try:
            stock = yf.Ticker(yf_code)
            info = stock.info
            
            # 提取有用的信息
            stock_data = {
                'code': code,
                'name': info.get('longName') or info.get('shortName'),
                
                # 基本信息
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                
                # 估值指标
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('trailingPegRatio'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'price_to_book': info.get('priceToBook'),
                'enterprise_to_revenue': info.get('enterpriseToRevenue'),
                'enterprise_to_ebitda': info.get('enterpriseToEbitda'),
                
                # 财务指标
                'profit_margins': info.get('profitMargins'),
                'gross_margins': info.get('grossMargins'),
                'operating_margins': info.get('operatingMargins'),
                'ebitda_margins': info.get('ebitdaMargins'),
                'return_on_assets': info.get('returnOnAssets'),
                'return_on_equity': info.get('returnOnEquity'),
                
                # 资产负债
                'total_cash': info.get('totalCash'),
                'total_debt': info.get('totalDebt'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'book_value': info.get('bookValue'),
                
                # 现金流
                'free_cashflow': info.get('freeCashflow'),
                'operating_cashflow': info.get('operatingCashflow'),
                
                # 增长指标
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
                
                # 股息指标
                'dividend_rate': info.get('dividendRate'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield'),
                
                # 风险指标
                'beta': info.get('beta'),
                
                # 股本信息
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'held_percent_insiders': info.get('heldPercentInsiders'),
                'held_percent_institutions': info.get('heldPercentInstitutions'),
                
                # 价格信息
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'fifty_day_average': info.get('fiftyDayAverage'),
                'two_hundred_day_average': info.get('twoHundredDayAverage'),
                
                # 成交量
                'average_volume': info.get('averageVolume'),
                'average_volume_10days': info.get('averageVolume10days'),
                
                # 更新时间
                'update_time': datetime.now()
            }
            
            return stock_data
            
        except Exception as e:
            print(f"获取{code}信息失败: {e}")
            return None
    
    def update_stock_info(self, code, stock_data):
        """
        更新股票信息到数据库
        
        参数:
            code: 股票代码
            stock_data: 股票信息字典
        """
        cursor = self.conn.cursor()
        
        # 构建SQL语句
        columns = ', '.join(stock_data.keys())
        placeholders = ', '.join(['?' for _ in stock_data])
        
        sql = f'''
            INSERT OR REPLACE INTO stock_info_extended 
            ({columns})
            VALUES ({placeholders})
        '''
        
        cursor.execute(sql, tuple(stock_data.values()))
        self.conn.commit()
    
    def update_all_stocks(self, limit=None, delay=QUEST_INTERVAL):
        """
        更新所有股票信息
        
        参数:
            limit: 限制更新数量（用于测试）
            delay: 每次请求间隔（秒）
        """
        self.connect_db()
        self.create_extended_table()
        
        stock_list = self.get_stock_list()
        total = len(stock_list)
        
        if limit:
            stock_list = stock_list[:limit]
            print(f"测试模式：仅更新前{limit}只股票")
        
        print(f"开始更新{len(stock_list)}只股票信息...")
        
        success_count = 0
        fail_count = 0
        
        for i, code in enumerate(stock_list, 1):
            print(f"[{i}/{len(stock_list)}] 正在更新 {code} ...")
            
            stock_data = self.fetch_stock_info(code)
            
            if stock_data:
                self.update_stock_info(code, stock_data)
                success_count += 1
                print(f"  ✓ 成功")
            else:
                fail_count += 1
                print(f"  ✗ 失败")
            
            # 延迟避免请求过快
            if i < len(stock_list):
                time.sleep(delay)
        
        self.close_db()
        
        print(f"\n更新完成！")
        print(f"成功: {success_count}")
        print(f"失败: {fail_count}")
        print(f"总计: {len(stock_list)}")


def main():
    """主函数"""
    updater = StockInfoUpdater()
    
    # 测试模式：仅更新前5只股票
    # updater.update_all_stocks(limit=5, delay=2)
    
    # 正式模式：更新所有股票
    updater.update_all_stocks(delay=1)


if __name__ == '__main__':
    """all"""
    main()


    """single"""
    # self=StockInfoUpdater()
    # code='600001'
    # stock_data = self.fetch_stock_info(code)
    # self.connect_db()
    # if stock_data:
    #     self.update_stock_info(code, stock_data)
    #     print(f"  ✓ 成功")
    # else:
    #     print(f"  ✗ 失败")
    # self.close_db()