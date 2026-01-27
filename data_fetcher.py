# 数据获取模块
import akshare as ak
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta
import config
from config import DATABASE_PATH,YEARS, RETRY_DELAYS, WORKERS_NUM
import os
import requests
import json

class DataFetcher:
    def __init__(self, use_proxy=config.USE_PROXY):
        # 确保使用项目根目录的数据库路径
        db_path = DATABASE_PATH
        self.conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        # 启用 WAL 模式以提高并发性能
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.init_database()
        
        # 代理配置
        self.use_proxy = use_proxy if use_proxy is not None else config.USE_PROXY
        self.proxy_api_url = config.PROXY_URL
        self.proxy_pool = []  # 代理池
        self.proxy_retry_delay = 6  # 代理失败后重新获取的延迟时间（秒）
        self.num_proxies = WORKERS_NUM  # 获取的代理数量
    
    def get_proxy_pool(self, num=None):
        """获取代理IP池
        
        Args:
            num: 获取的代理数量，默认使用WORKERS_NUM
        
        Returns:
            list: 代理字典列表
        """
        if num is None:
            num = self.num_proxies
            
        try:
            # 构建请求URL
            url = self.proxy_api_url.replace("{num}", str(num))
            
            # 获取代理IP
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                result = response.json()
                
                # 检查返回码
                if result.get('code') == 0 and 'data' in result:
                    proxy_data = result['data']
                    proxy_list_raw = proxy_data.get('proxy_list', [])
                    
                    if proxy_list_raw and len(proxy_list_raw) > 0:
                        proxy_list = []
                        # 设置默认过期时间为1小时后
                        expire_time = datetime.now() + timedelta(hours=1)
                        
                        for proxy_str in proxy_list_raw:
                            # 解析代理字符串格式: IP:端口:用户名:密码
                            parts = proxy_str.split(':')
                            if len(parts) == 4:
                                ip, port, username, password = parts
                                proxy_address = f"{ip}:{port}"
                                # 构建带认证的代理URL
                                proxy_url = f"http://{username}:{password}@{ip}:{port}"
                                
                                proxy_dict = {
                                    'address': proxy_address,
                                    'http': proxy_url,
                                    'https': proxy_url,
                                    'expire_time': expire_time
                                }
                                proxy_list.append(proxy_dict)
                        
                        self.proxy_pool = proxy_list
                        print(f"获取代理IP池成功: {len(proxy_list)} 个代理")
                        for i, proxy in enumerate(proxy_list, 1):
                            print(f"  代理{i}: {proxy['address']}, 有效期至: {proxy['expire_time'].strftime('%H:%M:%S')}")
                        
                        return proxy_list
            
            print("获取代理IP池失败")
            return []
            
        except Exception as e:
            print(f"获取代理IP池异常: {e}")
            return []
    
    def get_valid_proxy(self):
        """从代理池中获取一个有效的代理"""
        # 清理过期的代理
        now = datetime.now()
        self.proxy_pool = [p for p in self.proxy_pool if p['expire_time'] > now]
        
        # 如果代理池为空，重新获取
        if not self.proxy_pool:
            print("代理池为空，重新获取...")
            self.get_proxy_pool()
        
        # 返回第一个代理（如果有的话）
        if self.proxy_pool:
            return self.proxy_pool[0]
        return None
    
    def set_proxy_env(self, proxy=None):
        """设置代理环境变量（用于akshare）
        
        Args:
            proxy: 指定的代理字典，如果为None则从代理池获取
        """
        if not self.use_proxy:
            return False
        
        if proxy is None:
            proxy = self.get_valid_proxy()
        
        if proxy:
            os.environ['HTTP_PROXY'] = proxy['http']
            os.environ['HTTPS_PROXY'] = proxy['https']
            return True
        return False
    
    def clear_proxy_env(self):
        """清除代理环境变量"""
        if 'HTTP_PROXY' in os.environ:
            del os.environ['HTTP_PROXY']
        if 'HTTPS_PROXY' in os.environ:
            del os.environ['HTTPS_PROXY']
    
    def request_with_proxy_retry(self, request_func, *args, **kwargs):
        """使用代理重试机制执行请求
        
        Args:
            request_func: 请求函数（如ak.stock_info_a_code_name）
            *args, **kwargs: 传递给请求函数的参数
        
        Returns:
            请求结果
        """
        if not self.use_proxy:
            # 不使用代理，直接请求
            return request_func(*args, **kwargs)
        
        # 确保代理池有足够的代理
        if not self.proxy_pool:
            self.get_proxy_pool()
        
        max_retry_rounds = 10  # 最多重新获取代理池的次数
        
        for retry_round in range(max_retry_rounds):
            # 如果不是第一轮，重新获取代理池
            if retry_round > 0:
                print(f"所有代理都失败，等待 {self.proxy_retry_delay} 秒后重新获取代理... (第 {retry_round + 1}/{max_retry_rounds} 轮)")
                time.sleep(self.proxy_retry_delay)
                new_proxies = self.get_proxy_pool()
                if not new_proxies:
                    print(f"第 {retry_round + 1} 轮获取代理池失败")
                    continue
            
            # 使用当前代理池中的每个代理重试
            current_pool = self.proxy_pool.copy()
            for i, proxy in enumerate(current_pool, 1):
                try:
                    if retry_round > 0:
                        print(f"使用新代理 {i}/{len(current_pool)}: {proxy['address']} (第 {retry_round + 1} 轮)")
                    # else:
                    #     print(f"尝试使用代理 {i}/{len(current_pool)}: {proxy['address']}")
                    
                    self.set_proxy_env(proxy)
                    result = request_func(*args, **kwargs)
                    
                    if retry_round > 0:
                        print(f"✓ 新代理 {proxy['address']} 请求成功")
                    # else:
                    #     print(f"✓ 代理 {proxy['address']} 请求成功")
                    
                    return result
                    
                except Exception as e:
                    print(f"✗ 代理 {proxy['address']} 请求失败: {e}")
                    self.clear_proxy_env()
                    
                    # 从代理池中移除失败的代理
                    if proxy in self.proxy_pool:
                        self.proxy_pool.remove(proxy)
                    
                    # 如果还有其他代理，继续尝试
                    if i < len(current_pool):
                        continue
                    # 如果是当前池的最后一个代理，跳出内层循环，进入下一轮重试
                    break
        
        # 所有重试轮次都失败，抛出异常
        raise Exception(f"所有代理重试均失败（共 {max_retry_rounds} 轮）")
    
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
            # 使用代理重试机制获取股票列表
            stock_list = self.request_with_proxy_retry(ak.stock_info_a_code_name)
            
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
            # 使用代理重试机制获取实时数据
            realtime_data = self.request_with_proxy_retry(ak.stock_zh_a_spot_em)
            stock_data = realtime_data[realtime_data['代码'] == symbol]
            return stock_data
        except Exception as e:
            print(f"获取{symbol}实时数据失败: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol, period="daily", start_date=None, end_date=None):
        """获取历史数据（带重试机制）
        
        重试策略：
        - 第1次失败：延迟1分钟后重试
        - 第2次失败：延迟5分钟后重试
        - 第3次失败：延迟30分钟后重试
        - 第4次失败：退出程序
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*YEARS)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        for attempt in range(len(RETRY_DELAYS)):
            try:
                # 使用代理重试机制获取历史数据
                hist_data = self.request_with_proxy_retry(
                    ak.stock_zh_a_hist,
                    symbol=symbol,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                return hist_data
            except Exception as e:
                error_msg = str(e)
                print(f"获取{symbol}历史数据失败 (尝试 {attempt + 1}/{len(RETRY_DELAYS)}): {error_msg}")
                
                # 如果是最后一次尝试，退出程序
                if attempt == len(RETRY_DELAYS)-1:
                    print(f"获取{symbol}历史数据失败，已达到最大重试次数，程序退出")
                    import sys
                    sys.exit(1)
                
                # 等待后重试
                delay = RETRY_DELAYS[attempt]
                delay_minutes = delay / 60
                print(f"将在 {delay_minutes} 分钟后重试...")
                time.sleep(delay)
        
        return pd.DataFrame()

    def get_stock_info(self,code=None):
        """
        从数据库获取股票基本信息

        Args:
            code: 股票代码，如果为None则返回所有股票信息

        Returns:
            DataFrame: 包含股票信息的数据
        """
        conn = sqlite3.connect(DATABASE_PATH)

        if code:
            query = "SELECT * FROM stock_info WHERE code = ?"
            df = pd.read_sql_query(query, conn, params=(code,))
        else:
            query = "SELECT * FROM stock_info"
            df = pd.read_sql_query(query, conn)

        conn.close()
        return df

    def update_stock_info(self, markets=['sh', 'sz_main']):
        """更新股票基本信息（包括市值、市盈率、市净率）
        
        Args:
            markets: 市场列表，默认['sh', 'sz_main']表示上证和深圳主板
        """
        print(f"开始获取股票实时行情数据...")
        
        try:
            # 使用代理重试机制获取实时行情数据
            realtime_data = self.request_with_proxy_retry(ak.stock_zh_a_spot_em)
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
                    (code, name, market_cap, pe_ratio, pb_ratio)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    stock['代码'], 
                    stock['名称'],
                    stock['总市值'] if pd.notna(stock['总市值']) else None,
                    stock['市盈率-动态'] if pd.notna(stock['市盈率-动态']) else None,
                    stock['市净率'] if pd.notna(stock['市净率']) else None
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
            SELECT MAX(update_time) FROM stock_info WHERE code = ?
        ''', (symbol,))
        result = cursor.fetchone()
        return result[0] if result[0] else None
    
    def get_last_update_time(self, symbol):
        """获取股票最后更新时间戳（用于判断是否需要更新当天数据）"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT update_time FROM stock_info WHERE code = ?
        ''', (symbol,))
        result = cursor.fetchone()
        return result[0] if result and result[0] else None
    
    def get_first_update_date(self, symbol):
        """获取股票最早更新日期"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MIN(date) FROM daily_data WHERE code = ?
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
            
            # 增量更新：检查起始日期和结束日期
            if incremental:
                last_date = self.get_last_update_date(symbol)
                first_date = self.get_first_update_date(symbol)
                
                # 计算期望的起始日期（根据配置的年数）
                expected_start_date = (datetime.now() - timedelta(days=365*YEARS)).strftime('%Y-%m-%d')
                current_date = datetime.now().strftime('%Y-%m-%d')
                now = datetime.now()
                today_15pm = now.replace(hour=15, minute=0, second=0, microsecond=0)
                
                if last_date and first_date:
                    # 精确比较日期
                    # 检查是否需要补充历史数据（起始日期晚于期望日期）
                    need_historical = first_date > expected_start_date
                    
                    # 检查是否需要更新最新数据
                    # 1. 如果最后日期早于今天，肯定需要更新
                    # 2. 如果最后日期是今天，检查更新时间是否在15:00之后
                    need_latest = False


                    if last_date == (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d') and now >= today_15pm:
                        spot_data = ak.stock_zh_a_spot_em()
                        existing_stock = self.get_stock_info()
                        
                        # 获取最新的实时数据并更新
                        for _, stock in existing_stock.iterrows():
                            code = stock['code']
                            stock_data = spot_data[spot_data['代码'] == code]
                            if not stock_data.empty:
                                # 更新日线数据表中的最新交易日数据
                                cursor.execute('''
                                    INSERT OR REPLACE INTO daily_data 
                                    (code, date, open, high, low, close, volume, amount, turnover_rate)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    code, 
                                    datetime.now().strftime('%Y-%m-%d'),
                                    float(stock_data.iloc[0]['开盘']) if pd.notna(stock_data.iloc[0]['开盘']) else None,
                                    float(stock_data.iloc[0]['最高']) if pd.notna(stock_data.iloc[0]['最高']) else None,
                                    float(stock_data.iloc[0]['最低']) if pd.notna(stock_data.iloc[0]['最低']) else None,
                                    float(stock_data.iloc[0]['收盘']) if pd.notna(stock_data.iloc[0]['收盘']) else None,
                                    float(stock_data.iloc[0]['成交量']) if pd.notna(stock_data.iloc[0]['成交量']) else None,
                                    float(stock_data.iloc[0]['成交额']) if pd.notna(stock_data.iloc[0]['成交额']) else None,
                                    float(stock_data.iloc[0]['换手率']) if pd.notna(stock_data.iloc[0]['换手率']) else None
                                ))
                        
                        # 更新stock_info的update_time
                        update_time = datetime.now()
                        cursor.execute('''
                            UPDATE stock_info SET update_time = ? WHERE code = ?
                        ''', (update_time, symbol))
                    elif last_date == current_date and now >= today_15pm:
                        # 今天且已过15:00，检查更新时间
                        cursor.execute('''
                            SELECT update_time FROM stock_info 
                            WHERE code = ?
                        ''', (symbol,))
                        result = cursor.fetchone()
                        if result and result[0]:
                            update_time = datetime.fromisoformat(result[0])
                            # 如果更新时间早于今天15:00，需要更新
                            if update_time < today_15pm:
                                need_latest = True
                        else:
                            # 没有更新时间记录，需要更新
                            need_latest = True
                    elif now<today_15pm and last_date == (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'):
                        print("今天未过15:00,暂缓更新")
                    elif now < today_15pm and last_date <= (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'):
                        need_latest = True

                    if config.BACKDATE:
                        if need_historical or need_latest:
                            # 一次性获取从期望起始日期到当前日期的完整数据
                            hist_start = expected_start_date.replace('-', '')
                            if datetime.now()>today_15pm:
                                hist_end = datetime.now().strftime('%Y%m%d')
                            else:
                                hist_end = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                            hist_data = self.get_historical_data(symbol, start_date=hist_start, end_date=hist_end)
                            
                            if not hist_data.empty:
                                # 获取数据库中已有的日期集合
                                cursor.execute('SELECT date FROM daily_data WHERE code = ?', (symbol,))
                                existing_dates = set(row[0] for row in cursor.fetchall())
                                
                                # 只插入缺失的数据
                                for _, row in hist_data.iterrows():
                                    if row['日期'] not in existing_dates:
                                        cursor.execute('''
                                            INSERT OR REPLACE INTO daily_data 
                                            (code, date, open, high, low, close, volume, amount, turnover_rate)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        ''', (
                                            symbol, row['日期'], row['开盘'], row['最高'],
                                            row['最低'], row['收盘'], row['成交量'], row['成交额'],
                                            row['换手率'] if '换手率' in row and pd.notna(row['换手率']) else None
                                        ))
                                
                                # 更新stock_info的update_time
                                update_time = datetime.now()
                                cursor.execute('''
                                    UPDATE stock_info SET update_time = ? WHERE code = ?
                                ''', (update_time, symbol))
                        else:
                            # 数据已是最新且完整，跳过
                            return
                    else:
                        if need_latest:
                            # 获取最新数据
                            latest_data = self.get_historical_data(symbol, start_date=last_date.replace('-', ''), end_date=current_date.replace('-', ''))
                            
                            if not latest_data.empty:
                                # 获取数据库中已有的日期集合
                                cursor.execute('SELECT date FROM daily_data WHERE code = ?', (symbol,))
                                existing_dates = set(row[0] for row in cursor.fetchall())
                                
                                # 只插入缺失的数据
                                for _, row in latest_data.iterrows():
                                    if row['日期'] not in existing_dates:
                                        cursor.execute('''
                                            INSERT OR REPLACE INTO daily_data 
                                            (code, date, open, high, low, close, volume, amount, turnover_rate)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        ''', (
                                            symbol, row['日期'], row['开盘'], row['最高'],
                                            row['最低'], row['收盘'], row['成交量'], row['成交额'],
                                            row['换手率'] if '换手率' in row and pd.notna(row['换手率']) else None
                                        ))
                else:
                    # 首次获取，获取完整历史数据
                    hist_data = self.get_historical_data(symbol)
                    
                    if not hist_data.empty:
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
                        
                        # 更新stock_info的update_time
                        update_time = datetime.now()
                        cursor.execute('''
                            UPDATE stock_info SET update_time = ? WHERE code = ?
                        ''', (update_time, symbol))
            else:
                # 全量更新
                hist_data = self.get_historical_data(symbol)
                
                if not hist_data.empty:
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
                    
                    # 更新stock_info的update_time
                    update_time = datetime.now()
                    cursor.execute('''
                        UPDATE stock_info SET update_time = ? WHERE code = ?
                    ''', (update_time, symbol))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"更新{symbol}数据失败: {e}")
    
    def get_stock_data(self, symbol, days=1):
        """从数据库获取股票数据"""
        query = '''
            SELECT * FROM daily_data 
            WHERE code = ? 
            ORDER BY date DESC 
            LIMIT ?
        '''
        df = pd.read_sql_query(query, self.conn, params=(symbol, days))
        return df.sort_values('date')
    
    def init_all_stocks_data(self, markets=['sh', 'sz_main'], incremental=False, workers=WORKERS_NUM):
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
            """工作线程函数 - 每个线程创建自己的数据库连接"""
            conn = None
            max_retries = 5
            retry_delay = 0.5  # 秒
            
            try:
                # 每个线程创建自己的数据库连接，增加超时时间
                conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
                # 启用 WAL 模式以提高并发性能
                conn.execute('PRAGMA journal_mode=WAL')
                cursor = conn.cursor()
                
                # 计算期望的起始日期（根据配置的年数）
                expected_start_date = (datetime.now() - timedelta(days=365*YEARS)).strftime('%Y-%m-%d')
                current_date = datetime.now().strftime('%Y-%m-%d')
                now = datetime.now()
                today_15pm = now.replace(hour=15, minute=0, second=0, microsecond=0)
                
                if incremental:
                    # 获取最早和最晚更新日期
                    cursor.execute('SELECT MIN(date), MAX(date) FROM daily_data WHERE code = ?', (stock_code,))
                    result = cursor.fetchone()
                    first_date = result[0] if result[0] else None
                    last_date = result[1] if result[1] else None
                    
                    if first_date and last_date:
                        # 精确比较日期
                        # 检查是否需要补充历史数据（起始日期晚于期望日期）
                        need_historical = first_date > expected_start_date
                        
                        # 检查是否需要更新最新数据
                        # 1. 如果最后日期早于今天，肯定需要更新
                        # 2. 如果最后日期是今天，检查更新时间是否在15:00之后
                        need_latest = False
                        if last_date < current_date:
                            need_latest = True
                        elif last_date == current_date and now >= today_15pm:
                            # 今天且已过15:00，检查更新时间
                            cursor.execute('''
                                SELECT update_time FROM stock_info 
                                WHERE code = ?
                            ''', (stock_code,))
                            result = cursor.fetchone()
                            if result and result[0]:
                                update_time = datetime.fromisoformat(result[0])
                                # 如果更新时间早于今天15:00，需要更新
                                if update_time < today_15pm:
                                    need_latest = True
                            else:
                                # 没有更新时间记录，需要更新
                                need_latest = True
                        
                        if need_historical or need_latest:
                            # 一次性获取从期望起始日期到当前日期的完整数据
                            hist_start = expected_start_date.replace('-', '')
                            hist_end = datetime.now().strftime('%Y%m%d')
                            hist_data = self.get_historical_data(stock_code, start_date=hist_start, end_date=hist_end)
                            
                            if not hist_data.empty:
                                # 获取数据库中已有的日期集合
                                cursor.execute('SELECT date FROM daily_data WHERE code = ?', (stock_code,))
                                existing_dates = set(row[0] for row in cursor.fetchall())
                                
                                # 使用重试机制只插入缺失的数据
                                for attempt in range(max_retries):
                                    try:
                                        for _, row in hist_data.iterrows():
                                            if row['日期'] not in existing_dates:
                                                cursor.execute('''
                                                    INSERT OR REPLACE INTO daily_data 
                                                    (code, date, open, high, low, close, volume, amount, turnover_rate)
                                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                                ''', (
                                                    stock_code, row['日期'], row['开盘'], row['最高'],
                                                    row['最低'], row['收盘'], row['成交量'], row['成交额'],
                                                    row['换手率'] if '换手率' in row and pd.notna(row['换手率']) else None
                                                ))
                                        
                                        # 更新stock_info的update_time
                                        update_time = datetime.now()
                                        cursor.execute('''
                                            UPDATE stock_info SET update_time = ? WHERE code = ?
                                        ''', (update_time, stock_code))
                                        
                                        conn.commit()
                                        break
                                    except sqlite3.OperationalError as e:
                                        if 'locked' in str(e) and attempt < max_retries - 1:
                                            time.sleep(retry_delay * (attempt + 1))
                                            continue
                                        raise
                        else:
                            # 数据已是最新且完整，跳过
                            return True, stock_code
                    else:
                        # 首次获取完整历史数据
                        hist_data = self.get_historical_data(stock_code)
                        
                        if not hist_data.empty:
                            # 使用重试机制写入数据
                            for attempt in range(max_retries):
                                try:
                                    for _, row in hist_data.iterrows():
                                        cursor.execute('''
                                            INSERT OR REPLACE INTO daily_data 
                                            (code, date, open, high, low, close, volume, amount, turnover_rate)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        ''', (
                                            stock_code, row['日期'], row['开盘'], row['最高'],
                                            row['最低'], row['收盘'], row['成交量'], row['成交额'],
                                            row['换手率'] if '换手率' in row and pd.notna(row['换手率']) else None
                                        ))
                                    
                                    # 更新stock_info的update_time
                                    update_time = datetime.now()
                                    cursor.execute('''
                                        UPDATE stock_info SET update_time = ? WHERE code = ?
                                    ''', (update_time, stock_code))
                                    
                                    conn.commit()
                                    break
                                except sqlite3.OperationalError as e:
                                    if 'locked' in str(e) and attempt < max_retries - 1:
                                        time.sleep(retry_delay * (attempt + 1))
                                        continue
                                    raise
                else:
                    # 全量更新
                    hist_data = self.get_historical_data(stock_code)
                    
                    if not hist_data.empty:
                        # 使用重试机制写入数据
                        for attempt in range(max_retries):
                            try:
                                for _, row in hist_data.iterrows():
                                    cursor.execute('''
                                        INSERT OR REPLACE INTO daily_data 
                                        (code, date, open, high, low, close, volume, amount, turnover_rate)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    ''', (
                                        stock_code, row['日期'], row['开盘'], row['最高'],
                                        row['最低'], row['收盘'], row['成交量'], row['成交额'],
                                        row['换手率'] if '换手率' in row and pd.notna(row['换手率']) else None
                                    ))
                                
                                # 更新stock_info的update_time
                                update_time = datetime.now()
                                cursor.execute('''
                                    UPDATE stock_info SET update_time = ? WHERE code = ?
                                ''', (update_time, stock_code))
                                
                                conn.commit()
                                break
                            except sqlite3.OperationalError as e:
                                if 'locked' in str(e) and attempt < max_retries - 1:
                                    time.sleep(retry_delay * (attempt + 1))
                                    continue
                                raise
                
                time.sleep(0.1)
                return True, stock_code
                
            except Exception as e:
                return False, f"{stock_code}: {e}"
            finally:
                if conn:
                    conn.close()
        
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