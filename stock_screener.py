# 股票筛选模块
import pandas as pd
import sqlite3
from datetime import datetime
from data_fetcher import DataFetcher
from technical_indicators import TechnicalIndicators
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
        
        if len(stock_data) < 30:
            return False, "数据不足"
        
        # 价格筛选
        latest_price = stock_data['close'].iloc[-1]
        if latest_price < self.criteria['min_price'] or latest_price > self.criteria['max_price']:
            return False, f"价格不符合条件: {latest_price}"
        
        # 换手率筛选
        if 'turnover_rate' in stock_data.columns:
            avg_turnover_rate = stock_data['turnover_rate'].tail(5).mean()
            if pd.isna(avg_turnover_rate) or avg_turnover_rate < self.criteria['min_turnover_rate']:
                return False, f"换手率不足: {avg_turnover_rate if not pd.isna(avg_turnover_rate) else 'N/A'}"
        else:
            # 如果没有换手率字段，使用成交量作为备选
            avg_volume = stock_data['volume'].tail(5).mean()
            min_volume = 50000  # 备选最小成交量
            if avg_volume < min_volume:
                return False, f"成交量不足: {avg_volume}"
        
        # 获取技术信号
        signals = self.tech_indicators.get_latest_signals(stock_data)
        if signals is None:
            return False, "技术指标计算失败"
        
        return True, signals
    
    def screen_stocks_by_strategy(self, strategy_name):
        """根据策略筛选股票"""
        candidates = self.get_candidate_stocks()
        results = []
        
        total_candidates = len(candidates)
        print(f"\n开始筛选，候选股票数量: {total_candidates}")
        print(f"策略: {strategy_name}")
        print("=" * 60)
        
        processed = 0
        passed_filter = 0
        matched_strategy = 0
        
        for idx, stock in candidates.iterrows():
            try:
                processed += 1
                
                # 应用技术指标筛选
                passed, signals_or_reason = self.apply_technical_filters(stock['code'])
                
                if passed:
                    passed_filter += 1
                    
                    # 根据不同策略进行筛选
                    strategy_matched = False
                    
                    if strategy_name == "golden_cross":
                        strategy_matched = self._check_golden_cross_strategy(signals_or_reason)
                    elif strategy_name == "oversold_rebound":
                        strategy_matched = self._check_oversold_rebound_strategy(signals_or_reason)
                    elif strategy_name == "breakout":
                        strategy_matched = self._check_breakout_strategy(signals_or_reason)
                    
                    if strategy_matched:
                        matched_strategy += 1
                        
                        # 获取当前价格和基本信息
                        stock_data = self.data_fetcher.get_stock_data(stock['code'], days=1)
                        current_price = stock_data['close'].iloc[-1] if not stock_data.empty else 0
                        
                        results.append({
                            'code': stock['code'],
                            'name': stock['name'],
                            'current_price': current_price,
                            'signals': signals_or_reason,
                            'strategy': strategy_name,
                            'market_cap': stock.get('market_cap', None),
                            'pe_ratio': stock.get('pe_ratio', None)
                        })
                
                # 进度显示
                if processed % 50 == 0 or processed == total_candidates:
                    progress = processed / total_candidates * 100
                    print(f"进度: {processed}/{total_candidates} ({progress:.1f}%) | "
                          f"通过筛选: {passed_filter} | 符合策略: {matched_strategy}")
                    
            except Exception as e:
                print(f"处理股票 {stock['code']} 时出错: {e}")
                continue
        
        print("=" * 60)
        print(f"筛选完成！共处理 {processed} 只股票")
        print(f"通过技术筛选: {passed_filter} 只")
        print(f"符合策略条件: {matched_strategy} 只")
        
        return results
    
    def _check_golden_cross_strategy(self, signals):
        """黄金交叉策略"""
        return (signals['ma_trend'] == 'bullish' and 
                signals['macd_signal'] == 'bullish' and
                signals['rsi_signal'] != 'overbought')
    
    def _check_oversold_rebound_strategy(self, signals):
        """超卖反弹策略"""
        return (signals['rsi_signal'] == 'oversold' and
                signals['kdj_signal'] == 'oversold' and
                signals['bb_signal'] == 'oversold')
    
    def _check_breakout_strategy(self, signals):
        """突破策略"""
        return (signals['ma_trend'] == 'bullish' and
                signals['bb_signal'] != 'overbought' and
                signals['macd_signal'] == 'bullish')
    
    def get_custom_screen_results(self, custom_conditions):
        """自定义筛选条件"""
        candidates = self.get_candidate_stocks()
        results = []
        
        total_candidates = len(candidates)
        print(f"\n开始自定义筛选，候选股票数量: {total_candidates}")
        print(f"筛选条件: {custom_conditions}")
        print("=" * 60)
        
        processed = 0
        matched = 0
        
        for idx, stock in candidates.iterrows():
            try:
                processed += 1
                
                passed, signals_or_reason = self.apply_technical_filters(stock['code'])
                
                if passed and self._check_custom_conditions(signals_or_reason, custom_conditions):
                    matched += 1
                    
                    # 获取当前价格
                    stock_data = self.data_fetcher.get_stock_data(stock['code'], days=1)
                    current_price = stock_data['close'].iloc[-1] if not stock_data.empty else 0
                    
                    results.append({
                        'code': stock['code'],
                        'name': stock['name'],
                        'current_price': current_price,
                        'signals': signals_or_reason,
                        'strategy': 'custom',
                        'market_cap': stock.get('market_cap', None),
                        'pe_ratio': stock.get('pe_ratio', None)
                    })
                
                # 进度显示
                if processed % 50 == 0 or processed == total_candidates:
                    progress = processed / total_candidates * 100
                    print(f"进度: {processed}/{total_candidates} ({progress:.1f}%) | 符合条件: {matched}")
                    
            except Exception as e:
                print(f"处理股票 {stock['code']} 时出错: {e}")
                continue
        
        print("=" * 60)
        print(f"筛选完成！共处理 {processed} 只股票，符合条件: {matched} 只")
        
        return results
    
    def _check_custom_conditions(self, signals, conditions):
        """检查自定义条件"""
        for condition in conditions:
            signal_type = condition['signal']
            expected_value = condition['value']
            
            if signal_type in signals:
                if signals[signal_type] != expected_value:
                    return False
            else:
                return False
        
        return True
    
    def generate_report(self, results, strategy_name):
        """生成筛选报告"""
        if not results:
            print(f"\n策略 '{strategy_name}' 未找到符合条件的股票")
            return
        
        print(f"\n{'=' * 80}")
        print(f"{strategy_name.upper()} 策略筛选结果")
        print(f"{'=' * 80}")
        print(f"找到 {len(results)} 只符合条件的股票\n")
        
        # 按市值排序（如果有市值数据）
        sorted_results = sorted(results, key=lambda x: x.get('market_cap', 0) or 0, reverse=True)
        
        for i, stock in enumerate(sorted_results, 1):
            print(f"\n【{i}】 {stock['code']} - {stock['name']}")
            print(f"    当前价格: ¥{stock.get('current_price', 0):.2f}")
            
            if stock.get('market_cap'):
                market_cap_yi = stock['market_cap'] / 100000000
                print(f"    市值: {market_cap_yi:.2f}亿")
            
            if stock.get('pe_ratio'):
                print(f"    市盈率: {stock['pe_ratio']:.2f}")
            
            signals = stock['signals']
            print(f"    技术信号:")
            print(f"      • 均线趋势: {signals['ma_trend']}")
            print(f"      • RSI: {signals['rsi_signal']}")
            print(f"      • MACD: {signals['macd_signal']}")
            print(f"      • 布林带: {signals['bb_signal']}")
            print(f"      • KDJ: {signals['kdj_signal']}")
            print("-" * 80)
        
        # 保存结果到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screen_results_{strategy_name}_{timestamp}.csv"
        
        # 准备导出数据
        export_data = []
        for stock in sorted_results:
            signals = stock['signals']
            export_data.append({
                '股票代码': stock['code'],
                '股票名称': stock['name'],
                '当前价格': stock.get('current_price', 0),
                '市值(亿)': stock.get('market_cap', 0) / 100000000 if stock.get('market_cap') else None,
                '市盈率': stock.get('pe_ratio'),
                '均线趋势': signals['ma_trend'],
                'RSI信号': signals['rsi_signal'],
                'MACD信号': signals['macd_signal'],
                '布林带信号': signals['bb_signal'],
                'KDJ信号': signals['kdj_signal'],
                '策略': strategy_name
            })
        
        df_results = pd.DataFrame(export_data)
        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n✓ 结果已保存到: {filename}")
        print(f"{'=' * 80}\n")
    
    def close(self):
        """关闭资源"""
        self.data_fetcher.close()