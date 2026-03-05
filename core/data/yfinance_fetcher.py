#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yfinance数据获取模块 - 作为akshare的备用数据源
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from config import QUEST_INTERVAL
from config.config import YEARS

class YFinanceFetcher:
    """使用yfinance获取股票数据的类"""
    
    def __init__(self):
        """初始化yfinance数据获取器"""
        self.session = None
    
    @staticmethod
    def convert_code_to_yfinance(code):
        """
        将A股代码转换为yfinance格式
        
        Args:
            code: A股代码，如 '600000'
            
        Returns:
            yfinance格式的代码，如 '600000.SS' (上交所) 或 '000001.SZ' (深交所)
        """
        if not isinstance(code, str):
            code = str(code)
            
        # 如果已经包含后缀，直接返回
        if '.' in code:
            return code
            
        if code.startswith('6'):
            # 上海证券交易所
            return f"{code}.SS"
        elif code.startswith(('0', '3')):
            # 深圳证券交易所（主板和创业板）
            return f"{code}.SZ"
        elif code.startswith(('8', '4')):
            # 北京证券交易所
            return f"{code}.BJ"
        else:
            return code
    
    @staticmethod
    def convert_yfinance_to_code(yf_code):
        """
        将yfinance格式转换回A股代码
        
        Args:
            yf_code: yfinance格式的代码，如 '600000.SS'
            
        Returns:
            A股代码，如 '600000'
        """
        return yf_code.split('.')[0] if '.' in yf_code else yf_code
    
    def get_historical_data(self, symbol, start_date=None, end_date=None, period="daily"):
        """
        获取历史数据

        Args:
            symbol: 股票代码（A股格式，如 '600000'）
            start_date: 开始日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'
            period: 周期，默认 'daily'（yfinance使用 '1d'）

        Returns:
            DataFrame: 包含历史数据的DataFrame，列名与akshare保持一致
        """
        try:
            # 转换代码格式
            yf_symbol = self.convert_code_to_yfinance(symbol)

            # 转换日期格式
            if start_date:
                if len(start_date) == 8:  # YYYYMMDD格式
                    start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            else:
                start_date = (datetime.now() - timedelta(days=365*YEARS)).strftime('%Y-%m-%d')

            if end_date:
                if len(end_date) == 8:  # YYYYMMDD格式
                    end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            else:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # 获取数据
            ticker = yf.Ticker(yf_symbol)
            hist_data = ticker.history(start=start_date, end=end_date, repair=True)

            if hist_data.empty:
                print(f"yfinance未获取到{symbol}的数据")
                return pd.DataFrame()

            # 转换为akshare格式的DataFrame
            result = pd.DataFrame({
                '日期': hist_data.index.strftime('%Y-%m-%d'),
                '开盘': hist_data['Open'].values,
                '最高': hist_data['High'].values,
                '最低': hist_data['Low'].values,
                '收盘': hist_data['Close'].values,
                '成交量': hist_data['Volume'].values,
                '成交额': (hist_data['Close'] * hist_data['Volume']).values,  # 估算成交额
                '换手率': None  # yfinance不提供换手率
            })

            return result

        except Exception as e:
            print(f"yfinance获取{symbol}历史数据失败: {e}")
            return pd.DataFrame()

    def get_stock_list(self, markets=['sh', 'sz_main']):
        """
        获取股票列表
        注意：yfinance不直接提供A股列表，此方法返回空DataFrame
        建议使用akshare获取股票列表

        Args:
            markets: 市场列表

        Returns:
            空DataFrame（yfinance不支持此功能）
        """
        print("⚠️ yfinance不支持获取A股股票列表，请使用akshare")
        return pd.DataFrame()

    def get_realtime_data(self, symbol):
        """
        获取实时行情数据

        Args:
            symbol: 股票代码（A股格式）

        Returns:
            DataFrame: 实时行情数据
        """
        try:
            yf_symbol = self.convert_code_to_yfinance(symbol)
            ticker = yf.Ticker(yf_symbol)

            # 获取最新的行情数据
            info = ticker.info
            hist = ticker.history(period='1d')

            if hist.empty:
                return pd.DataFrame()

            # 构建类似akshare的实时数据格式
            result = pd.DataFrame({
                '代码': [symbol],
                '名称': [info.get('longName', info.get('shortName', ''))],
                '最新价': [hist['Close'].iloc[-1]],
                '涨跌幅': [info.get('regularMarketChangePercent', 0)],
                '涨跌额': [info.get('regularMarketChange', 0)],
                '成交量': [hist['Volume'].iloc[-1]],
                '成交额': [hist['Close'].iloc[-1] * hist['Volume'].iloc[-1]],
                '开盘': [hist['Open'].iloc[-1]],
                '最高': [hist['High'].iloc[-1]],
                '最低': [hist['Low'].iloc[-1]],
                '昨收': [info.get('previousClose', 0)],
                '总市值': [info.get('marketCap', None)],
                '市盈率-动态': [info.get('trailingPE', None)],
                '市净率': [info.get('priceToBook', None)]
            })

            return result

        except Exception as e:
            print(f"yfinance获取{symbol}实时数据失败: {e}")
            return pd.DataFrame()



if __name__ == "__main__":
    # 测试代码
    print("测试yfinance数据获取...")
    fetcher = YFinanceFetcher()

    # 测试获取历史数据
    test_symbol = "600000"  # 浦发银行
    print(f"\n测试获取{test_symbol}的历史数据...")
    hist_data = fetcher.get_historical_data(
        test_symbol,
        start_date="20240101",
        end_date="20240131"
    )

    if not hist_data.empty:
        print(f"成功获取{len(hist_data)}条数据")
        print(hist_data.head())
    else:
        print("获取数据失败")
    
    # 测试获取实时数据
    print(f"\n测试获取{test_symbol}的实时数据...")
    realtime_data = fetcher.get_realtime_data(test_symbol)
    if not realtime_data.empty:
        print("成功获取实时数据")
        print(realtime_data)
    else:
        print("获取实时数据失败")
