# 数据获取模块
import akshare as ak
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta
from config import DATABASE_PATH,YEARS


import os

class DataFetcher:
    def __init__(self):
        # 确保使用项目根目录的数据库路径
        db_path = DATABASE_PATH
        self.conn = sqlite3.connect(db_path)
        self.init_database()
    
    def init_database(self):
        """初始化数据库表"""
        cursor = self.conn.cursor()
        
        # 股票基本信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_info (
                code TEXT PRIMARY KEY,
                name TEXT,
                market_cap REAL,
                pe_ratio REAL,
                pb_ratio REAL,
                update_time TIMESTAMP
            )
        ''')
        
        # 日线数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_data (
                code TEXT,
                date DATE,
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
        
        self.conn.commit()
    
    def get_stock_list(self, markets=['sh', 'sz_main']):
        """获取A股股票列表
        
        Args:
            markets: 市场列表，默认['sh', 'sz_main']表示上证和深圳主板
                    'sh' - 上海证券交易所 (60xxxx)
                    'sz_main' - 深圳主板 (00xxxx)
                    'sz_gem' - 创业板 (30xxxx)
                    'bj' - 北京证券交易所 (8xxxxx, 4xxxxx)
        """
        try:
            # 获取沪深A股列表
            stock_list = ak.stock_info_a_code_name()
            
            # 根据市场代码过滤
            if markets:
                filtered_list = []
                for _, stock in stock_list.iterrows():
                    code = stock['code']
                    # 上证: 60开头
                    if 'sh' in markets and code.startswith('60'):
                        filtered_list.append(stock)
                    # 深圳主板: 00开头
                    elif 'sz_main' in markets and code.startswith('00'):
                        filtered_list.append(stock)
                    # 创业板: 30开头
                    elif 'sz_gem' in markets and code.startswith('30'):
                        filtered_list.append(stock)
                    # 北交所: 8和4开头
                    elif 'bj' in markets and (code.startswith('8') or code.startswith('4')):
                        filtered_list.append(stock)
                
                stock_list = pd.DataFrame(filtered_list)
            
            return stock_list
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol):
        """获取实时行情数据"""
        try:
            # 获取实时数据
            realtime_data = ak.stock_zh_a_spot_em()
            stock_data = realtime_data[realtime_data['代码'] == symbol]
            return stock_data
        except Exception as e:
            print(f"获取{symbol}实时数据失败: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol, period="daily", start_date=None, end_date=None):
        """获取历史数据"""
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*YEARS)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            # 获取历史数据
            hist_data = ak.stock_zh_a_hist(symbol=symbol, period=period, 
                                         start_date=start_date, end_date=end_date)
            return hist_data
        except Exception as e:
            print(f"获取{symbol}历史数据失败: {e}")
            return pd.DataFrame()
    
    def update_stock_info(self, markets=['sh', 'sz_main']):
        """更新股票基本信息（包括市值、市盈率、市净率）
        
        Args:
            markets: 市场列表，默认['sh', 'sz_main']表示上证和深圳主板
        """
        print(f"开始获取股票实时行情数据...")
        
        try:
            # 获取实时行情数据（包含市值、市盈率、市净率）
            realtime_data = ak.stock_zh_a_spot_em()
            print(f"成功获取 {len(realtime_data)} 只股票的实时数据")
        except Exception as e:
            print(f"获取实时行情数据失败: {e}")
            return
        
        # 根据市场过滤
        if markets:
            filtered_data = []
            for _, stock in realtime_data.iterrows():
                code = stock['代码']
                # 上证: 60开头
                if 'sh' in markets and code.startswith('60'):
                    filtered_data.append(stock)
                # 深圳主板: 00开头
                elif 'sz_main' in markets and code.startswith('00'):
                    filtered_data.append(stock)
                # 创业板: 30开头
                elif 'sz_gem' in markets and code.startswith('30'):
                    filtered_data.append(stock)
                # 北交所: 8和4开头
                elif 'bj' in markets and (code.startswith('8') or code.startswith('4')):
                    filtered_data.append(stock)
            
            realtime_data = pd.DataFrame(filtered_data)
        
        if realtime_data.empty:
            print("未获取到符合条件的股票数据")
            return
        
        cursor = self.conn.cursor()
        update_time = datetime.now()
        total_count = len(realtime_data)
        
        print(f"开始写入 {total_count} 只股票的基本信息...")
        
        for count, (idx, stock) in enumerate(realtime_data.iterrows(), 1):
            try:
                # 写入基本信息（包括市值、市盈率、市净率）
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_info 
                    (code, name, market_cap, pe_ratio, pb_ratio, update_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    stock['代码'], 
                    stock['名称'],
                    stock['总市值'] if pd.notna(stock['总市值']) else None,
                    stock['市盈率-动态'] if pd.notna(stock['市盈率-动态']) else None,
                    stock['市净率'] if pd.notna(stock['市净率']) else None,
                    update_time
                ))
                
                # 进度显示
                if count % 100 == 0:
                    print(f"已写入 {count}/{total_count} 只股票基本信息")
                    self.conn.commit()  # 批量提交
                
            except Exception as e:
                print(f"写入{stock['代码']}信息失败: {e}")
                continue
        
        self.conn.commit()
        print(f"股票基本信息写入完成（包含市值、市盈率、市净率）")
    
    def get_last_update_date(self, symbol):
        """获取股票最后更新日期"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(date) FROM daily_data WHERE code = ?
        ''', (symbol,))
        result = cursor.fetchone()
        return result[0] if result[0] else None
    
    def update_daily_data(self, symbol, incremental=True):
        """更新单只股票的日线数据
        
        Args:
            symbol: 股票代码
            incremental: 是否增量更新，True表示只更新最新数据
        """
        try:
            cursor = self.conn.cursor()
            
            # 增量更新：只获取最后更新日期之后的数据
            if incremental:
                last_date = self.get_last_update_date(symbol)
                if last_date:
                    # 从最后日期的下一天开始更新
                    start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y%m%d')
                    end_date = datetime.now().strftime('%Y%m%d')
                    
                    # 如果已经是最新，跳过
                    if start_date >= end_date:
                        return
                    
                    hist_data = self.get_historical_data(symbol, start_date=start_date, end_date=end_date)
                else:
                    # 首次获取，获取一年数据
                    hist_data = self.get_historical_data(symbol)
            else:
                # 全量更新
                hist_data = self.get_historical_data(symbol)
            
            if hist_data.empty:
                return
            
            # 批量写入数据
            for _, row in hist_data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO daily_data 
                    (code, date, open, high, low, close, volume, amount, turnover_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, row['日期'], row['开盘'], row['最高'],
                    row['最低'], row['收盘'], row['成交量'], row['成交额'],
                    row['换手率'] if '换手率' in row and pd.notna(row['换手率']) else None
                ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"更新{symbol}数据失败: {e}")
    
    def get_stock_data(self, symbol, days=1000):
        """从数据库获取股票数据"""
        query = '''
            SELECT * FROM daily_data 
            WHERE code = ? 
            ORDER BY date DESC 
            LIMIT ?
        '''
        df = pd.read_sql_query(query, self.conn, params=(symbol, days))
        return df.sort_values('date')
    
    def init_all_stocks_data(self, markets=['sh', 'sz_main'], incremental=False, workers=4):
        """初始化所有股票数据（并行获取，批量写入）
        
        Args:
            markets: 市场列表，默认['sh', 'sz_main']表示上证和深圳主板
            incremental: 是否增量更新，默认False表示首次初始化用全量模式
            workers: 并行工作线程数，默认4
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock
        
        print(f"开始初始化股票数据（市场: {', '.join(markets)}，并行度: {workers}）...")
        
        # 1. 先更新股票基本信息
        self.update_stock_info(markets=markets)
        
        # 2. 获取股票列表
        stock_list = self.get_stock_list(markets=markets)
        if stock_list.empty:
            print("未获取到股票列表")
            return
        
        total_stocks = len(stock_list)
        update_mode = '增量' if incremental else '全量'
        print(f"\n开始{update_mode}更新 {total_stocks} 只股票的日线数据...")
        
        success_count = 0
        fail_count = 0
        lock = Lock()
        
        def update_stock_worker(stock_code):
            """工作线程函数"""
            try:
                self.update_daily_data(stock_code, incremental=incremental)
                time.sleep(0.1)  # 减少延迟（并行后可以降低）
                return True, stock_code
            except Exception as e:
                return False, f"{stock_code}: {e}"
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            futures = {executor.submit(update_stock_worker, stock['code']): stock['code'] 
                      for _, stock in stock_list.iterrows()}
            
            # 处理完成的任务
            for count, future in enumerate(as_completed(futures), 1):
                success, result = future.result()
                
                with lock:
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        print(f"更新失败: {result}")
                    
                    # 进度显示
                    if count % 50 == 0:
                        print(f"进度: {count}/{total_stocks} (成功: {success_count}, 失败: {fail_count})")
        
        print(f"\n数据初始化完成！")
        print(f"总计: {total_stocks} 只股票")
        print(f"成功: {success_count} 只")
        print(f"失败: {fail_count} 只")
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()