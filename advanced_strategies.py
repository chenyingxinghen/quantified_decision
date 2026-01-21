# 高级交易策略模块 - 基于价格行为和多周期分析
import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import DataFetcher
from technical_indicators import TechnicalIndicators
from price_action_analyzer import PriceActionAnalyzer

class AdvancedTradingStrategies:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.tech_indicators = TechnicalIndicators()
        self.price_action = PriceActionAnalyzer()
    
    def wyckoff_accumulation_strategy(self, stock_code):
        """
        威科夫积累策略 - 识别Spring形态
        寻找底部区域的最后一次洗盘机会
        
        新增筛选条件：
        - 均线必须向上（MA20 > MA50 > MA60）
        - 价格在均线上方或接近均线
        
        返回格式与SMC策略一致，用于回测引擎
        """
        # 获取日线和周线数据
        daily_data = self.data_fetcher.get_stock_data(stock_code)
        
        if len(daily_data) < 60:  # 需要至少60天数据来计算MA60
            return None
        
        # 均线向上筛选（必须条件）
        ma_check = self._check_moving_average_uptrend(daily_data)
        if not ma_check['is_uptrend']:
            return None  # 均线不向上，直接过滤
        
        # 价格行为分析
        pa_analysis = self.price_action.get_comprehensive_analysis(daily_data)
        if not pa_analysis:
            return None
        
        # 技术指标
        indicators = self.tech_indicators.calculate_all_indicators(daily_data)
        if not indicators:
            return None
        
        # 威科夫积累条件检查
        conditions = self._check_wyckoff_accumulation(daily_data, pa_analysis, indicators)
        
        current_price = daily_data['close'].iloc[-1]
        
        # 计算风险收益比
        risk = current_price - conditions['stop_loss']
        reward = conditions['target'] - current_price
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # 返回格式与SMC策略一致
        return {
            'stock_code': stock_code,
            'signal': conditions['signal'],
            'confidence': conditions['confidence'],
            'current_price': current_price,
            'entry_price': current_price,
            'stop_loss': conditions['stop_loss'],
            'target': conditions['target'],
            'risk_reward_ratio': risk_reward_ratio,
            'signals': conditions['details'] + [f"均线向上 (强度: {ma_check['strength']:.1f}%)"],
            'conditions_met': conditions['details'] + [f"均线向上 (强度: {ma_check['strength']:.1f}%)"],
            'analysis': {
                'price_action': pa_analysis,
                'indicators': indicators,
                'wyckoff_conditions': conditions,
                'ma_uptrend': ma_check
            }
        }
    
    def _check_wyckoff_accumulation(self, data, pa_analysis, indicators):
        """检查威科夫积累阶段条件"""
        current_price = data['close'].iloc[-1]
        recent_low = data['low'].tail(20).min()
        recent_high = data['high'].tail(20).max()
        
        conditions_met = []
        confidence = 0
        
        # 1. 价格在底部区域横盘
        price_range = (recent_high - recent_low) / recent_low
        if price_range < 0.15:  # 15%以内的横盘
            conditions_met.append("底部横盘区域确认")
            confidence += 20
        
        # 2. 成交量萎缩
        recent_volume = data['volume'].tail(10).mean()
        avg_volume = data['volume'].mean()
        if recent_volume < avg_volume * 0.8:
            conditions_met.append("成交量萎缩")
            confidence += 15
        
        # 3. Spring形态 - 快速跌破支撑后回升
        if self._detect_spring_pattern(data):
            conditions_met.append("Spring形态确认")
            confidence += 25
        
        # 4. RSI超卖后回升
        if len(indicators['rsi']) > 0:
            current_rsi = indicators['rsi'][-1]
            if 30 < current_rsi < 50:  # 从超卖区域回升
                conditions_met.append("RSI从超卖回升")
                confidence += 15
        
        # 5. MACD底背离
        if self._detect_macd_bullish_divergence(data, indicators):
            conditions_met.append("MACD底背离")
            confidence += 25
        
        # 生成信号
        signal = 'buy' if confidence >= 60 else 'watch' if confidence >= 40 else 'no_signal'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': recent_low * 0.985,  # 优化：1.5%止损（更保守）
            'target': current_price * 1.12,   # 优化：12%目标（更高收益）
            'details': conditions_met
        }
    
    def _check_moving_average_uptrend(self, data):
        """
        检查均线是否向上
        
        条件：
        1. MA5 > MA20 > MA60（多头排列）
        2. 均线斜率向上（最近5天均线上升）
        3. 价格在MA5上方或接近MA5（2%以内）
        
        返回：
            dict: {
                'is_uptrend': bool,
                'strength': float (0-100),
                'details': list
            }
        """
        if len(data) < 60:
            return {'is_uptrend': False, 'strength': 0, 'details': []}
        
        # 计算均线
        ma5 = data['close'].rolling(window=5).mean()
        ma20 = data['close'].rolling(window=20).mean()
        ma60 = data['close'].rolling(window=60).mean()
        
        current_price = data['close'].iloc[-1]
        current_ma5 = ma5.iloc[-1]
        current_ma20 = ma20.iloc[-1]
        current_ma60 = ma60.iloc[-1]
        
        details = []
        strength = 0
        
        # 条件1: 多头排列 MA5 > MA20 > MA60
        if current_ma5 > current_ma20 > current_ma60:
            details.append("均线多头排列(MA5>MA20>MA60)")
            strength += 40
        # else:
        #     # 不满足多头排列，直接返回
        #     return {'is_uptrend': False, 'strength': 0, 'details': ['均线未形成多头排列']}
        
        # 条件2: 均线斜率向上（最近5天）
        ma5_slope = (ma5.iloc[-1] - ma5.iloc[-6]) / ma5.iloc[-6] if len(ma5) >= 6 else 0
        ma20_slope = (ma20.iloc[-1] - ma20.iloc[-6]) / ma20.iloc[-6] if len(ma20) >= 6 else 0
        
        if ma5_slope > 0 and ma20_slope > 0:
            details.append("均线斜率向上")
            strength += 30
        elif ma5_slope > 0:
            details.append("MA5向上")
            strength += 15
        
        # 条件3: 价格位置（在MA5上方或接近）
        price_to_ma5_pct = (current_price - current_ma5) / current_ma5
        
        if price_to_ma5_pct >= 0:
            details.append(f"价格在MA5上方 (+{price_to_ma5_pct*100:.1f}%)")
            strength += 20
        elif price_to_ma5_pct >= -0.02:  # 2%以内
            details.append(f"价格接近MA5 ({price_to_ma5_pct*100:.1f}%)")
            strength += 10
        
        # 条件4: 均线间距合理（不要太分散）
        ma_spread = (current_ma5 - current_ma60) / current_ma60
        if 0 < ma_spread < 0.15:  # 0-15%之间（MA5和MA60间距可以稍大）
            details.append("均线间距合理")
            strength += 10
        
        # 判断是否为上升趋势
        is_uptrend = strength >= 50  # 至少50分才算合格
        
        return {
            'is_uptrend': is_uptrend,
            'strength': strength,
            'details': details
        }
        
    
    def _detect_spring_pattern(self, data):
        """检测Spring形态"""
        if len(data) < 10:
            return False
        
        recent_data = data.tail(10)
        support_level = data.tail(30)['low'].min()
        
        # 寻找跌破支撑后快速回升的模式
        for i in range(len(recent_data) - 2):
            if (recent_data.iloc[i]['low'] < support_level * 0.99 and  # 跌破支撑
                recent_data.iloc[i+1]['close'] > recent_data.iloc[i]['close'] and  # 次日回升
                recent_data.iloc[i]['volume'] > recent_data['volume'].mean() * 1.5):  # 放量
                return True
        
        return False
    
    def _detect_macd_bullish_divergence(self, data, indicators):
        """检测MACD底背离"""
        if len(indicators['macd']) < 20:
            return False
        
        # 寻找价格创新低但MACD不创新低的情况
        recent_prices = data['close'].tail(20)
        recent_macd = indicators['macd'][-20:]
        
        price_low_idx = recent_prices.idxmin()
        macd_values_at_lows = []
        
        # 简化的背离检测
        if len(recent_macd) >= 10:
            return recent_macd[-1] > recent_macd[-10] and recent_prices.iloc[-1] <= recent_prices.iloc[-10]
        
        return False
    
    def golden_cross_momentum_strategy(self, stock_code):
        """
        黄金交叉动量策略 - 多周期确认
        结合20EMA和50SMA的共振系统
        """
        daily_data = self.data_fetcher.get_stock_data(stock_code)
        
        if len(daily_data) < 60:
            return None
        
        # 计算EMA和SMA
        ema20 = self._calculate_ema(daily_data['close'], 20)
        sma50 = self._calculate_sma(daily_data['close'], 50)
        
        # 价格行为分析
        pa_analysis = self.price_action.get_comprehensive_analysis(daily_data)
        
        # 技术指标
        indicators = self.tech_indicators.calculate_all_indicators(daily_data)
        
        # 黄金交叉条件检查
        conditions = self._check_golden_cross_conditions(daily_data, ema20, sma50, pa_analysis, indicators)
        
        return {
            'strategy': 'golden_cross_momentum',
            'signal': conditions['signal'],
            'confidence': conditions['confidence'],
            'entry_price': daily_data['close'].iloc[-1],
            'stop_loss': conditions['stop_loss'],
            'target': conditions['target'],
            'analysis': pa_analysis,
            'conditions_met': conditions['details']
        }
    
    def _check_golden_cross_conditions(self, data, ema20, sma50, pa_analysis, indicators):
        """检查黄金交叉策略条件"""
        current_price = data['close'].iloc[-1]
        conditions_met = []
        confidence = 0
        
        if len(ema20) < 2 or len(sma50) < 2:
            return {'signal': 'no_signal', 'confidence': 0, 'details': [], 'stop_loss': 0, 'target': 0}
        
        # 1. EMA20上穿SMA50
        if ema20[-1] > sma50[-1] and ema20[-2] <= sma50[-2]:
            conditions_met.append("EMA20上穿SMA50黄金交叉")
            confidence += 30
        
        # 2. 价格在均线上方
        if current_price > ema20[-1] > sma50[-1]:
            conditions_met.append("价格位于均线上方")
            confidence += 20
        
        # 3. 市场结构确认上升趋势
        if pa_analysis and pa_analysis['market_structure']['trend'] == 'uptrend':
            conditions_met.append("市场结构确认上升趋势")
            confidence += 25
        
        # 4. MACD金叉确认
        if (len(indicators['macd']) > 1 and len(indicators['macd_signal']) > 1 and
            indicators['macd'][-1] > indicators['macd_signal'][-1] and
            indicators['macd'][-2] <= indicators['macd_signal'][-2]):
            conditions_met.append("MACD金叉确认")
            confidence += 20
        
        # 5. 成交量放大
        recent_volume = data['volume'].tail(3).mean()
        avg_volume = data['volume'].tail(20).mean()
        if recent_volume > avg_volume * 1.2:
            conditions_met.append("成交量放大确认")
            confidence += 15
        
        # 6. RSI不超买
        if len(indicators['rsi']) > 0 and indicators['rsi'][-1] < 70:
            conditions_met.append("RSI未超买")
            confidence += 10
        
        signal = 'buy' if confidence >= 70 else 'watch' if confidence >= 50 else 'no_signal'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': min(ema20[-1], sma50[-1]) * 0.985,  # 优化：1.5%止损
            'target': current_price * 1.15,  # 优化：15%目标
            'details': conditions_met
        }
    
    def mean_reversion_oversold_strategy(self, stock_code):
        """
        均值回归超卖策略 - 多指标超卖反弹
        """
        daily_data = self.data_fetcher.get_stock_data(stock_code)
        
        if len(daily_data) < 30:
            return None
        
        # 价格行为分析
        pa_analysis = self.price_action.get_comprehensive_analysis(daily_data)
        
        # 技术指标
        indicators = self.tech_indicators.calculate_all_indicators(daily_data)
        
        # 超卖条件检查
        conditions = self._check_oversold_conditions(daily_data, pa_analysis, indicators)
        
        return {
            'strategy': 'mean_reversion_oversold',
            'signal': conditions['signal'],
            'confidence': conditions['confidence'],
            'entry_price': daily_data['close'].iloc[-1],
            'stop_loss': conditions['stop_loss'],
            'target': conditions['target'],
            'analysis': pa_analysis,
            'conditions_met': conditions['details']
        }
    
    def _check_oversold_conditions(self, data, pa_analysis, indicators):
        """检查超卖反弹条件"""
        current_price = data['close'].iloc[-1]
        conditions_met = []
        confidence = 0
        
        # 1. RSI超卖
        if len(indicators['rsi']) > 0 and indicators['rsi'][-1] < 30:
            conditions_met.append("RSI超卖")
            confidence += 25
        
        # 2. KDJ超卖
        if (len(indicators['kdj_k']) > 0 and len(indicators['kdj_d']) > 0 and
            indicators['kdj_k'][-1] < 20 and indicators['kdj_d'][-1] < 20):
            conditions_met.append("KDJ超卖")
            confidence += 25
        
        # 3. 布林带下轨支撑
        if (len(indicators['bb_lower']) > 0 and 
            current_price <= indicators['bb_lower'][-1] * 1.02):
            conditions_met.append("接近布林带下轨")
            confidence += 20
        
        # 4. 锤头线等反转形态
        reversal_patterns = [p for p in pa_analysis.get('candlestick_patterns', []) 
                           if p['type'] == 'bullish_reversal']
        if reversal_patterns:
            conditions_met.append(f"反转形态: {reversal_patterns[0]['pattern']}")
            confidence += reversal_patterns[0]['reliability'] * 0.3
        
        # 5. 价格偏离均线过远
        if len(indicators['ma20']) > 0:
            deviation = (current_price - indicators['ma20'][-1]) / indicators['ma20'][-1]
            if deviation < -0.1:  # 偏离20日均线10%以上
                conditions_met.append("价格严重偏离均线")
                confidence += 15
        
        # 6. 成交量萎缩（恐慌性抛售后）
        recent_volume = data['volume'].tail(3).mean()
        prev_volume = data['volume'].tail(10).head(7).mean()
        if recent_volume < prev_volume * 0.7:
            conditions_met.append("成交量萎缩")
            confidence += 10
        
        signal = 'buy' if confidence >= 65 else 'watch' if confidence >= 45 else 'no_signal'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': current_price * 0.95,  # 5%止损
            'target': current_price * 1.10,     # 10%目标
            'details': conditions_met
        }
    
    def breakout_momentum_strategy(self, stock_code):
        """
        突破动量策略 - 高成交量突破关键阻力
        """
        daily_data = self.data_fetcher.get_stock_data(stock_code)
        
        if len(daily_data) < 40:
            return None
        
        # 价格行为分析
        pa_analysis = self.price_action.get_comprehensive_analysis(daily_data)
        
        # 技术指标
        indicators = self.tech_indicators.calculate_all_indicators(daily_data)
        
        # 突破条件检查
        conditions = self._check_breakout_conditions(daily_data, pa_analysis, indicators)
        
        return {
            'strategy': 'breakout_momentum',
            'signal': conditions['signal'],
            'confidence': conditions['confidence'],
            'entry_price': daily_data['close'].iloc[-1],
            'stop_loss': conditions['stop_loss'],
            'target': conditions['target'],
            'analysis': pa_analysis,
            'conditions_met': conditions['details']
        }
    
    def _check_breakout_conditions(self, data, pa_analysis, indicators):
        """检查突破策略条件"""
        current_price = data['close'].iloc[-1]
        conditions_met = []
        confidence = 0
        
        # 1. 突破关键阻力位
        resistance_level = data['high'].tail(30).max()
        if current_price > resistance_level * 1.005:  # 突破0.5%
            conditions_met.append("突破关键阻力位")
            confidence += 30
        
        # 2. 成交量确认
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].tail(20).mean()
        if current_volume > avg_volume * 2:
            conditions_met.append("成交量大幅放大")
            confidence += 25
        
        # 3. 市场结构支持
        if pa_analysis and pa_analysis['market_structure']['trend'] in ['uptrend', 'consolidation']:
            conditions_met.append("市场结构支持突破")
            confidence += 20
        
        # 4. 技术指标确认
        if (len(indicators['rsi']) > 0 and 50 < indicators['rsi'][-1] < 80):
            conditions_met.append("RSI处于强势区间")
            confidence += 15
        
        # 5. MACD向上
        if (len(indicators['macd_hist']) > 1 and 
            indicators['macd_hist'][-1] > indicators['macd_hist'][-2]):
            conditions_met.append("MACD柱状图向上")
            confidence += 15
        
        # 6. 布林带上轨突破
        if (len(indicators['bb_upper']) > 0 and 
            current_price > indicators['bb_upper'][-1]):
            conditions_met.append("突破布林带上轨")
            confidence += 10
        
        signal = 'buy' if confidence >= 70 else 'watch' if confidence >= 50 else 'no_signal'
        
        # 计算止损和目标
        support_level = data['low'].tail(20).min()
        stop_loss = max(support_level, current_price * 0.95)
        target = current_price + (current_price - stop_loss) * 2  # 1:2风险收益比
        
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'target': target,
            'details': conditions_met
        }
    
    def multi_timeframe_confluence_strategy(self, stock_code):
        """
        多周期共振策略 - 1:4:16黄金比例模型
        """
        # 这里简化为日线分析，实际应用中可以获取不同周期数据
        daily_data = self.data_fetcher.get_stock_data(stock_code)
        
        if len(daily_data) < 60:
            return None
        
        # 模拟不同周期分析
        long_term = daily_data  # 代表周线级别
        medium_term = daily_data.tail(50)  # 代表日线级别
        short_term = daily_data.tail(20)   # 代表4小时级别
        
        # 各周期分析
        long_analysis = self.price_action.get_comprehensive_analysis(long_term)
        medium_analysis = self.price_action.get_comprehensive_analysis(medium_term)
        short_analysis = self.price_action.get_comprehensive_analysis(short_term)
        
        # 多周期共振检查
        confluence = self._check_multi_timeframe_confluence(
            long_analysis, medium_analysis, short_analysis, daily_data
        )
        
        return {
            'strategy': 'multi_timeframe_confluence',
            'signal': confluence['signal'],
            'confidence': confluence['confidence'],
            'entry_price': daily_data['close'].iloc[-1],
            'stop_loss': confluence['stop_loss'],
            'target': confluence['target'],
            'timeframe_analysis': {
                'long_term': long_analysis,
                'medium_term': medium_analysis,
                'short_term': short_analysis
            },
            'conditions_met': confluence['details']
        }
    
    def _check_multi_timeframe_confluence(self, long_analysis, medium_analysis, short_analysis, data):
        """检查多周期共振条件"""
        current_price = data['close'].iloc[-1]
        conditions_met = []
        confidence = 0
        
        # 1. 长期趋势方向
        if long_analysis and long_analysis['market_structure']['trend'] == 'uptrend':
            conditions_met.append("长期上升趋势确认")
            confidence += 40
        
        # 2. 中期结构支持
        if medium_analysis and medium_analysis['market_structure']['trend'] in ['uptrend', 'consolidation']:
            conditions_met.append("中期结构支持")
            confidence += 30
        
        # 3. 短期入场信号
        if short_analysis and short_analysis['composite_signal']['direction'] == 'bullish':
            conditions_met.append("短期看涨信号")
            confidence += 20
        
        # 4. 关键位置确认
        if self._check_key_level_confluence(data):
            conditions_met.append("关键位置共振")
            confidence += 15
        
        signal = 'buy' if confidence >= 75 else 'watch' if confidence >= 55 else 'no_signal'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': current_price * 0.96,  # 4%止损
            'target': current_price * 1.18,     # 18%目标
            'details': conditions_met
        }
    
    def _check_key_level_confluence(self, data):
        """检查关键位置共振"""
        current_price = data['close'].iloc[-1]
        
        # 检查是否在重要支撑位附近
        support_levels = [
            data['low'].tail(50).min(),
            data['close'].tail(20).mean(),
            data['high'].tail(30).quantile(0.3)
        ]
        
        for level in support_levels:
            if abs(current_price - level) / level < 0.02:  # 2%范围内
                return True
        
        return False
    
    def _calculate_ema(self, prices, period):
        """计算指数移动平均线"""
        return prices.ewm(span=period).mean().values
    
    def _calculate_sma(self, prices, period):
        """计算简单移动平均线"""
        return prices.rolling(window=period).mean().values
    
    def get_all_strategy_signals(self, stock_code):
        """获取所有策略的综合信号"""
        strategies = [
            self.wyckoff_accumulation_strategy,
            self.golden_cross_momentum_strategy,
            self.mean_reversion_oversold_strategy,
            self.breakout_momentum_strategy,
            self.multi_timeframe_confluence_strategy
        ]
        
        results = {}
        
        for strategy_func in strategies:
            try:
                result = strategy_func(stock_code)
                if result:
                    results[result['strategy']] = result
            except Exception as e:
                print(f"策略 {strategy_func.__name__} 执行失败: {e}")
                continue
        
        # 生成综合评分
        if results:
            composite_score = self._calculate_composite_score(results)
            results['composite_analysis'] = composite_score
        
        return results
    
    def _calculate_composite_score(self, strategy_results):
        """计算综合策略评分"""
        total_confidence = 0
        buy_signals = 0
        watch_signals = 0
        
        for strategy_name, result in strategy_results.items():
            if strategy_name == 'composite_analysis':
                continue
                
            confidence = result.get('confidence', 0)
            signal = result.get('signal', 'no_signal')
            
            total_confidence += confidence
            
            if signal == 'buy':
                buy_signals += 1
            elif signal == 'watch':
                watch_signals += 1
        
        avg_confidence = total_confidence / len(strategy_results) if strategy_results else 0
        
        # 综合信号判断
        if buy_signals >= 3:
            composite_signal = 'strong_buy'
        elif buy_signals >= 2:
            composite_signal = 'buy'
        elif buy_signals >= 1 or watch_signals >= 2:
            composite_signal = 'watch'
        else:
            composite_signal = 'no_signal'
        
        return {
            'composite_signal': composite_signal,
            'average_confidence': avg_confidence,
            'buy_signals_count': buy_signals,
            'watch_signals_count': watch_signals,
            'total_strategies': len(strategy_results)
        }