#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合数据获取模块 - 支持akshare和yfinance自动切换
当akshare失败时自动切换到yfinance作为备用数据源
"""
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
import config
from config import DATABASE_PATH, YEARS, RETRY_DELAYS, WORKERS_NUM

# 导入akshare数据获取器
try:
    from .data_fetcher import DataFetcher
    AKSHARE_AVAILABLE = True
except ImportError:
    print("⚠️ 无法导入akshare数据获取器")
    AKSHARE_AVAILABLE = False

# 导入yfinance数据获取器
try:
    from .yfinance_fetcher import YFinanceFetcher
    YFINANCE_AVAILABLE = True
except ImportError:
    print("⚠️ 无法导入yfinance数据获取器，请安装: pip install yfinance")
    YFINANCE_AVAILABLE = False


class HybridDataFetcher:
    """
    混合数据获取器
    优先使用akshare，失败时自动切换到yfinance
    """
    
    def __init__(self, use_proxy=config.USE_PROXY, prefer_source='yfinance'):
        """
        初始化混合数据获取器
        
        Args:
            use_proxy: 是否使用代理（仅对akshare有效）
            prefer_source: 优先使用的数据源，'akshare' 或 'yfinance'
        """
        self.prefer_source = prefer_source
        self.current_source = prefer_source
        
        # 初始化数据库连接
        self.conn = sqlite3.connect(DATABASE_PATH, timeout=30.0, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')
        
        # 初始化数据获取器
        self.akshare_fetcher = None
        self.yfinance_fetcher = None
        
        if AKSHARE_AVAILABLE:
            try:
                self.akshare_fetcher = DataFetcher(use_proxy=use_proxy)
                print("✓ akshare数据获取器初始化成功")
            except Exception as e:
                print(f"⚠️ akshare数据获取器初始化失败: {e}")
        
        if YFINANCE_AVAILABLE:
            try:
                self.yfinance_fetcher = YFinanceFetcher()
                print("✓ yfinance数据获取器初始化成功")
            except Exception as e:
                print(f"⚠️ yfinance数据获取器初始化失败: {e}")
        
        # 统计信息
        self.stats = {
            'akshare_success': 0,
            'akshare_fail': 0,
            'yfinance_success': 0,
            'yfinance_fail': 0
        }
    
    def get_historical_data(self, symbol, period="daily", start_date=None, end_date=None,fallback=True):
        """
        获取历史数据（支持自动切换数据源）
        
        Args:
            symbol: 股票代码
            period: 周期
            start_date: 开始日期
            end_date: 结束日期
            fallback: 是否在失败时切换到备用数据源
            
        Returns:
            DataFrame: 历史数据
        """
        # 首先尝试优先数据源
        if self.prefer_source == 'akshare' and self.akshare_fetcher:
            try:
                print(f"使用akshare获取{symbol}的历史数据...")
                data = self.akshare_fetcher.get_historical_data(
                    symbol, period=period, start_date=start_date, end_date=end_date
                )
                if not data.empty:
                    self.stats['akshare_success'] += 1
                    self.current_source = 'akshare'
                    return data
                else:
                    self.stats['akshare_fail'] += 1
            except Exception as e:
                print(f"akshare获取失败: {e}")
                self.stats['akshare_fail'] += 1
        
        elif self.prefer_source == 'yfinance' and self.yfinance_fetcher:
            try:
                print(f"使用yfinance获取{symbol}的历史数据...")
                data = self.yfinance_fetcher.get_historical_data(
                    symbol, start_date=start_date, end_date=end_date
                )
                if not data.empty:
                    self.stats['yfinance_success'] += 1
                    self.current_source = 'yfinance'
                    return data
                else:
                    self.stats['yfinance_fail'] += 1
            except Exception as e:
                print(f"yfinance获取失败: {e}")
                self.stats['yfinance_fail'] += 1
        
        # 如果优先数据源失败且允许切换，尝试备用数据源
        if fallback:
            if self.prefer_source == 'akshare' and self.yfinance_fetcher:
                print(f"切换到yfinance作为备用数据源...")
                try:
                    data = self.yfinance_fetcher.get_historical_data(
                        symbol, start_date=start_date, end_date=end_date
                    )
                    if not data.empty:
                        self.stats['yfinance_success'] += 1
                        self.current_source = 'yfinance'
                        print(f"✓ yfinance备用数据源获取成功")
                        return data
                    else:
                        self.stats['yfinance_fail'] += 1
                except Exception as e:
                    print(f"yfinance备用数据源也失败: {e}")
                    self.stats['yfinance_fail'] += 1
            
            elif self.prefer_source == 'yfinance' and self.akshare_fetcher:
                print(f"切换到akshare作为备用数据源...")
                try:
                    data = self.akshare_fetcher.get_historical_data(
                        symbol, period=period, start_date=start_date, end_date=end_date
                    )
                    if not data.empty:
                        self.stats['akshare_success'] += 1
                        self.current_source = 'akshare'
                        print(f"✓ akshare备用数据源获取成功")
                        return data
                    else:
                        self.stats['akshare_fail'] += 1
                except Exception as e:
                    print(f"akshare备用数据源也失败: {e}")
                    self.stats['akshare_fail'] += 1
        
        print(f"所有数据源均失败，无法获取{symbol}的数据")
        return pd.DataFrame()
    
    def get_stock_list(self, markets=['sh', 'sz_main']):
        """
        获取股票列表（优先使用akshare）
        
        Args:
            markets: 市场列表
            
        Returns:
            DataFrame: 股票列表
        """
        if self.akshare_fetcher:
            try:
                return self.akshare_fetcher.get_stock_list(markets=markets)
            except Exception as e:
                print(f"akshare获取股票列表失败: {e}")
        
        print("⚠️ yfinance不支持获取A股股票列表")
        return pd.DataFrame()
    
    def get_realtime_data(self, symbol):
        """
        获取实时数据（支持自动切换）
        
        Args:
            symbol: 股票代码
            
        Returns:
            DataFrame: 实时数据
        """
        # 优先使用akshare
        if self.akshare_fetcher:
            try:
                data = self.akshare_fetcher.get_realtime_data(symbol)
                if not data.empty:
                    return data
            except Exception as e:
                print(f"akshare获取实时数据失败: {e}")
        
        # 备用yfinance
        if self.yfinance_fetcher:
            try:
                print("切换到yfinance获取实时数据...")
                data = self.yfinance_fetcher.get_realtime_data(symbol)
                if not data.empty:
                    return data
            except Exception as e:
                print(f"yfinance获取实时数据失败: {e}")
        
        return pd.DataFrame()
    
    def update_daily_data(self, symbol, incremental=True):
        """
        更新单只股票的日线数据（使用混合数据源）
        
        Args:
            symbol: 股票代码
            incremental: 是否增量更新
        """
        try:
            cursor = self.conn.cursor()
            
            # 获取数据库中的最后更新日期
            if incremental:
                last_date = self._get_last_update_date(symbol)
                first_date = self._get_first_update_date(symbol)
                
                expected_start_date = (datetime.now() - timedelta(days=365*YEARS)).strftime('%Y-%m-%d')
                current_date = datetime.now().strftime('%Y-%m-%d')
                
                if last_date and first_date:
                    need_historical = first_date > expected_start_date
                    need_latest = last_date < current_date
                    
                    if not need_historical and not need_latest:
                        return  # 数据已是最新
                    
                    # 确定获取数据的日期范围
                    if need_historical:
                        start_date = expected_start_date.replace('-', '')
                    else:
                        start_date = last_date.replace('-', '')
                    
                    end_date = (datetime.now()+timedelta(days=1)).strftime('%Y-%m-%d').replace('-', '')
                else:
                    # 首次获取
                    start_date = expected_start_date.replace('-', '')
                    end_date = current_date.replace('-', '')
            else:
                # 全量更新
                start_date = None
                end_date = None
            
            # 使用混合数据源获取数据
            hist_data = self.get_historical_data(
                symbol, 
                start_date=start_date, 
                end_date=end_date
            )
            
            if not hist_data.empty:
                # 写入数据库
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
                print(f"✓ 更新{symbol}数据成功（数据源: {self.current_source}）")
            else:
                print(f"✗ 未能获取{symbol}的数据")
                
        except Exception as e:
            print(f"更新{symbol}数据失败: {e}")
    
    def _get_last_update_date(self, symbol):
        """获取股票最后更新日期"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT MAX(date) FROM daily_data WHERE code = ?', (symbol,))
        result = cursor.fetchone()
        return result[0] if result[0] else None
    
    def _get_first_update_date(self, symbol):
        """获取股票最早更新日期"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT MIN(date) FROM daily_data WHERE code = ?', (symbol,))
        result = cursor.fetchone()
        return result[0] if result[0] else None
    
    def print_stats(self):
        """打印数据源使用统计"""
        print("\n" + "="*50)
        print("数据源使用统计:")
        print(f"  akshare成功: {self.stats['akshare_success']} 次")
        print(f"  akshare失败: {self.stats['akshare_fail']} 次")
        print(f"  yfinance成功: {self.stats['yfinance_success']} 次")
        print(f"  yfinance失败: {self.stats['yfinance_fail']} 次")
        
        total_success = self.stats['akshare_success'] + self.stats['yfinance_success']
        total_fail = self.stats['akshare_fail'] + self.stats['yfinance_fail']
        total = total_success + total_fail
        
        if total > 0:
            success_rate = (total_success / total) * 100
            print(f"  总成功率: {success_rate:.2f}%")
        print("="*50 + "\n")
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # 测试混合数据获取器
    print("测试混合数据获取器...")
    
    fetcher = HybridDataFetcher(prefer_source='akshare')
    
    # 测试获取历史数据
    test_symbol = "600000"
    print(f"\n测试获取{test_symbol}的历史数据...")
    hist_data = fetcher.get_historical_data(
        test_symbol,
        start_date="20240101",
        end_date="20240131"
    )
    
    if not hist_data.empty:
        print(f"成功获取{len(hist_data)}条数据")
        print(hist_data.head())
    
    # 打印统计信息
    fetcher.print_stats()
    
    # 关闭连接
    fetcher.close()
