# SMC流动性猎取策略 V2 - 基于机构订单流的量化改进版
# 核心改进：置信度与胜率强相关，避免100%置信度陷阱

import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import DataFetcher
from technical_indicators import TechnicalIndicators
from trend_line_analyzer import TrendLineAnalyzer
import talib

class SMCLiquidityStrategy:
    """
    SMC策略V2 - 核心改进点：
    
    1. 必须条件筛选：只有满足核心条件才给信号
    2. 置信度=胜率映射：基于历史回测数据校准
    3. 机构足迹识别：流动性猎取+订单块+FVG三重验证
    4. 入场时机严格控制：避免追高
    """
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.tech_indicators = TechnicalIndicators()
        self.trend_analyzer = TrendLineAnalyzer(lookback_days=120)  # 使用6个月历史数据
        
        # 策略参数
        self.params = {
            'lookback_period': 90,
            'liquidity_sweep_threshold': 0.02,
            'spring_recovery_bars': 5,
            'order_block_strength': 1.5,
            'fvg_min_gap_ratio': 0.001,
            
            'trend_ma_long_period': 60,
            'trend_ma_mid_period': 20,
            'trend_ma_short_period': 5,  # 短均线周期'
            
            'volume_surge_ratio': 1.2,
            'atr_period': 10,
            'min_entry_quality': 85,   # 最低入场质量要求
            'min_confidence':80,       # 默认70
            'min_trend_strength': 75,  # 最低趋势强度要求（新增）
        }
    
    def screen_stock(self, stock_code):
        """
        核心筛选逻辑 - 基于SMC理论的严格实现 + 趋势线分析
        
        必须条件（缺一不可）：
        1. 大趋势向上（价格 > MA50，MA50向上）
        2. 至少一个核心信号（流动性猎取/订单块/FVG）
        3. 入场时机合理（不追高）+ 趋势线位置良好
        4. 风险收益比 >= 2:1
        
        新增：趋势线分析
        - 必须接近支撑趋势线（入场质量 >= 60）
        - 避免在通道上方或阻力位附近买入
        """
        try:
            data = self.data_fetcher.get_stock_data(stock_code, days=300)
            
            if len(data) < 60:
                return None
            
            current_price = data['close'].iloc[-1]
            
            # 计算ATR
            atr = talib.ATR(data['high'].values, data['low'].values, 
                        data['close'].values, timeperiod=self.params['atr_period'])[-1]
            
            # ========== 第一步：必须条件检查 ==========
            
            # 1. 大趋势必须向上
            market_structure = self._check_market_structure_strict(data)
            if not market_structure['is_uptrend']:
                return None  # 直接过滤，不给信号
            
            # 2. 趋势线分析（新增）- 放宽条件
            trend_analysis = self.trend_analyzer.analyze(data)
            
            # 入场质量必须 >= min_entry_quality（默认0，完全放开）
            if trend_analysis['entry_quality'] < self.params['min_entry_quality']:
                return None
            
            # 趋势强度必须 >= min_trend_strength（默认0，完全放开）
            if trend_analysis['trend_strength'] < self.params['min_trend_strength']:
                return None
            
            # 只过滤明显在通道上方的情况（放宽阻力位检查）
            if trend_analysis['current_position'] == 'above_channel':
                return None
            
            # 放宽支撑跌破检查：只有明显跌破才过滤
            # 如果有核心信号，即使轻微跌破也可以接受
            if trend_analysis['broken_support']:
                # 检查跌破程度
                if trend_analysis['analysis'].get('distance_from_support_pct', 0) < -5:
                    # 跌破超过5%才真正过滤
                    return None
            
            # 3. 检测核心SMC信号（放宽要求）
            liquidity_grab = self._detect_liquidity_grab_strict(data)
            order_block = self._identify_order_block_strict(data)
            fvg = self._detect_fvg_strict(data)
            
            # 至少有一个核心信号（保持不变）
            core_signals = []
            if liquidity_grab['detected']:
                core_signals.append('liquidity_grab')
            if order_block['found']:
                core_signals.append('order_block')
            if fvg['detected']:
                core_signals.append('fvg')
            
            # 如果没有核心信号，但市场结构很强，也可以给信号（新增）
            if len(core_signals) == 0:
                # 检查是否有强市场结构 + 好的趋势线位置
                if (market_structure.get('conditions_met', 0) >= 3 and 
                    trend_analysis['entry_quality'] >= 70 and
                    trend_analysis['current_position'] == 'near_support'):
                    # 创建一个虚拟的"趋势线支撑"信号
                    core_signals.append('trend_support')
                else:
                    return None  # 没有任何信号，直接过滤
            
            # 4. 入场时机检查（放宽）
            timing_check = self._check_entry_timing_strict(data, current_price)
            # 不再强制要求时机完美，只作为置信度参考
            # if not timing_check['is_good']:
            #     return None  # 时机不对，直接过滤
            
            # ========== 第二步：计算置信度（基于信号质量） ==========
            
            confidence_score = self._calculate_confidence_v2(
                market_structure, liquidity_grab, order_block, fvg,
                timing_check, trend_analysis, data
            )
            
            # 置信度必须 >= min_confidence（默认0，完全放开）
            if confidence_score < self.params['min_confidence']:
                return None
            
            # ========== 第三步：计算止损和目标 ==========
            
            stop_loss = self._calculate_stop_loss(
                current_price, liquidity_grab, order_block, fvg, atr, trend_analysis
            )
            
            target = self._calculate_target(current_price, stop_loss, atr)
            
            # 5. 风险收益比必须 >= 1.5:1（放宽）
            risk = current_price - stop_loss
            reward = target - current_price
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            if risk_reward_ratio < 1.5:
                return None  # 风险收益比不够，过滤
            
            # ========== 第四步：生成信号和详细说明 ==========
            
            signals = self._generate_signal_details(
                market_structure, liquidity_grab, order_block, fvg,
                timing_check, trend_analysis, confidence_score
            )
            
            # 信号分级
            if confidence_score >= 85 and len(core_signals) >= 2 and trend_analysis['entry_quality'] >= 80:
                signal = 'strong_buy'
            elif confidence_score >= 70:
                signal = 'buy'
            else:
                signal = 'watch'
            
            return {
                'stock_code': stock_code,
                'signal': signal,
                'confidence': confidence_score,
                'current_price': current_price,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'target': target,
                'risk_reward_ratio': risk_reward_ratio,
                'atr': atr,
                'signals': signals,
                'core_signals': core_signals,
                'analysis': {
                    'liquidity_grab': liquidity_grab,
                    'order_block': order_block,
                    'fvg': fvg,
                    'market_structure': market_structure,
                    'timing': timing_check,
                    'trend_analysis': trend_analysis
                }
            }
            
        except Exception as e:
            print(f"分析股票 {stock_code} 时出错: {e}")
            return None
    
    def _check_market_structure_strict(self, data):
        """
        多周期趋势验证 - 避免将反弹误判为上升趋势
        
        核心改进：
        1. 三周期均线系统：长期(90)、中期(30)、短期(5)
        2. 大趋势验证：长期均线必须向上
        3. 多周期一致性：短中长期趋势方向一致
        4. 反弹识别：区分趋势转换和短期反弹
        5. 趋势强度评分：综合评估趋势质量
        """
        if len(data) < self.params['trend_ma_long_period']:
            return {'is_uptrend': False, 'trend_strength': 0}
        
        # 计算三条均线
        ma_long = talib.SMA(data['close'].values, timeperiod=self.params['trend_ma_long_period'])
        ma_mid = talib.SMA(data['close'].values, timeperiod=self.params['trend_ma_mid_period'])
        ma_short = talib.SMA(data['close'].values, timeperiod=self.params['trend_ma_short_period'])
        current_price = data['close'].iloc[-1]
        
        # ========== 第一步：大趋势验证（长期均线） ==========
        
        # 长期均线斜率（必须向上）
        ma_long_slope = (ma_long[-1] - ma_long[-20]) / ma_long[-20] if ma_long[-20] > 0 else 0
        long_trend_up = ma_long_slope > 0  # 长期趋势向上
        
        # 价格相对长期均线位置
        price_vs_long = (current_price - ma_long[-1]) / ma_long[-1]
        above_long_ma = price_vs_long > -0.03  # 允许轻微跌破3%
        
        # ========== 第二步：多周期一致性检查 ==========
        
        # 中期均线斜率
        ma_mid_slope = (ma_mid[-1] - ma_mid[-10]) / ma_mid[-10] if ma_mid[-10] > 0 else 0
        mid_trend_up = ma_mid_slope > -0.01  # 允许轻微下降
        
        # 短期均线斜率
        ma_short_slope = (ma_short[-1] - ma_short[-3]) / ma_short[-3] if ma_short[-3] > 0 else 0
        short_trend_up = ma_short_slope > -0.02  # 短期允许更大波动
        
        # 均线排列（多头排列）
        ma_alignment = ma_short[-1] > ma_mid[-1] * 0.98 and ma_mid[-1] > ma_long[-1] * 0.98
        
        # ========== 第三步：反弹识别机制 ==========
        
        # 检查是否是下降趋势中的反弹
        is_bounce_in_downtrend = False
        
        if not long_trend_up:  # 长期趋势不向上
            # 检查价格是否短期反弹
            recent_low = data['low'].tail(10).min()
            bounce_strength = (current_price - recent_low) / recent_low
            
            # 如果短期反弹超过5%但长期趋势向下，标记为反弹
            if bounce_strength > 0.05:
                is_bounce_in_downtrend = True
        
        # ========== 第四步：HH-HL结构验证 ==========
        
        recent_data = data.tail(30)
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_data) - 2):
            if (recent_data.iloc[i]['high'] > recent_data.iloc[i-1]['high'] and
                recent_data.iloc[i]['high'] > recent_data.iloc[i+1]['high']):
                swing_highs.append(recent_data.iloc[i]['high'])
            
            if (recent_data.iloc[i]['low'] < recent_data.iloc[i-1]['low'] and
                recent_data.iloc[i]['low'] < recent_data.iloc[i+1]['low']):
                swing_lows.append(recent_data.iloc[i]['low'])
        
        has_hh_hl = False
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            has_hh_hl = (swing_highs[-1] > swing_highs[-2] and 
                        swing_lows[-1] > swing_lows[-2])
        
        # ========== 第五步：趋势强度评分（0-100） ==========
        
        trend_strength = 0
        
        # 1. 长期趋势方向（30分）
        if long_trend_up:
            if ma_long_slope > 0.05:  # 强劲上升
                trend_strength += 30
            elif ma_long_slope > 0.02:
                trend_strength += 25
            elif ma_long_slope > 0:
                trend_strength += 20
        
        # 2. 多周期一致性（25分）
        consistency_score = sum([long_trend_up, mid_trend_up, short_trend_up])
        if consistency_score == 3:
            trend_strength += 25
        elif consistency_score == 2:
            trend_strength += 15
        elif consistency_score == 1:
            trend_strength += 5
        
        # 3. 均线排列（20分）
        if ma_alignment:
            trend_strength += 20
        elif ma_short[-1] > ma_mid[-1] or ma_mid[-1] > ma_long[-1]:
            trend_strength += 10
        
        # 4. HH-HL结构（15分）
        if has_hh_hl:
            trend_strength += 15
        
        # 5. 价格位置（10分）
        if price_vs_long > 0.05:  # 价格远高于长期均线
            trend_strength += 10
        elif price_vs_long > 0:
            trend_strength += 7
        elif price_vs_long > -0.03:
            trend_strength += 3
        
        # 反弹惩罚：如果是下降趋势中的反弹，大幅降低强度
        if is_bounce_in_downtrend:
            trend_strength = min(trend_strength * 0.5, 40)  # 最多40分
        
        # ========== 第六步：综合判断 ==========
        
        # 严格的上升趋势定义：
        # 1. 长期趋势必须向上（核心条件）
        # 2. 至少两个周期趋势一致
        # 3. 不是下降趋势中的反弹
        # 4. 趋势强度 >= 40
        
        is_uptrend = (
            long_trend_up and  # 长期趋势向上（必须）
            consistency_score >= 2 and  # 至少两个周期一致
            not is_bounce_in_downtrend and  # 不是反弹
            trend_strength >= 40  # 趋势强度足够
        )
        
        return {
            'is_uptrend': is_uptrend,
            'trend_strength': trend_strength,
            'ma_long': ma_long[-1],
            'ma_mid': ma_mid[-1],
            'ma_short': ma_short[-1],
            'ma_long_slope': ma_long_slope,
            'ma_mid_slope': ma_mid_slope,
            'ma_short_slope': ma_short_slope,
            'long_trend_up': long_trend_up,
            'mid_trend_up': mid_trend_up,
            'short_trend_up': short_trend_up,
            'consistency_score': consistency_score,
            'ma_alignment': ma_alignment,
            'has_hh_hl': has_hh_hl,
            'is_bounce': is_bounce_in_downtrend,
            'price_vs_long_pct': price_vs_long * 100,
            'conditions_met': consistency_score + (1 if ma_alignment else 0) + (1 if has_hh_hl else 0),
            'strength': 'strong' if trend_strength >= 70 else ('moderate' if trend_strength >= 50 else 'weak')
        }

    
    def _detect_liquidity_grab_strict(self, data):
        """
        严格的流动性猎取检测
        
        核心逻辑（威科夫Spring）：
        1. 跌破近期明显低点（扫止损）
        2. 快速收回且收盘在支撑上方（假跌破）
        3. 伴随成交量放大（机构吸筹）
        4. 形成长下影线（锤子线）
        5. 之后没有再次跌破
        
        只有同时满足才返回detected=True
        """
        if len(data) < self.params['lookback_period']:
            return {'detected': False}
        
        recent_data = data.tail(self.params['lookback_period'])
        
        # 找到前期支撑位（排除最近3根K线）
        support_low = recent_data['low'].iloc[:-3].min()
        
        # 检查最近N根K线（使用spring_recovery_bars参数）
        for i in range(1, min(self.params['spring_recovery_bars'] + 2, len(data))):
            sweep_bar = data.iloc[-(i+1)]
            recovery_bar = data.iloc[-i]
            
            # 条件1：跌破支撑（使用liquidity_sweep_threshold参数）
            swept = sweep_bar['low'] < support_low * (1 - self.params['liquidity_sweep_threshold'])
            
            # 条件2：快速收回（收盘回到支撑上方）
            recovered = recovery_bar['close'] > support_low * (1 + self.params['liquidity_sweep_threshold'] * 0.5)
            
            # 条件3：成交量放大（使用volume_surge_ratio参数）
            avg_volume = recent_data['volume'].mean()
            volume_surge = sweep_bar['volume'] > avg_volume * self.params['volume_surge_ratio']
            
            # 条件4：长下影线
            body = abs(sweep_bar['close'] - sweep_bar['open'])
            lower_shadow = min(sweep_bar['open'], sweep_bar['close']) - sweep_bar['low']
            has_shadow = lower_shadow > body * 1.5 if body > 0 else lower_shadow > 0
            
            # 条件5：之后未再次跌破
            no_retest = all(data.iloc[j]['low'] > support_low * 0.997 
                          for j in range(max(0, len(data)-i), len(data)))
            
            if swept and recovered and volume_surge and has_shadow and no_retest:
                # 计算质量分数（0-100）
                quality = 0
                
                # 收回力度（30分）
                recovery_strength = (recovery_bar['close'] - sweep_bar['low']) / (support_low - sweep_bar['low'])
                quality += min(recovery_strength * 30, 30)
                
                # 成交量确认（25分）
                vol_ratio = sweep_bar['volume'] / avg_volume
                if vol_ratio > 2.5:
                    quality += 25
                elif vol_ratio > 2.0:
                    quality += 20
                elif vol_ratio > self.params['volume_surge_ratio']:
                    quality += 15
                
                # 下影线质量（20分）
                shadow_ratio = lower_shadow / body if body > 0 else 10
                if shadow_ratio > 3:
                    quality += 20
                elif shadow_ratio > 2:
                    quality += 15
                elif shadow_ratio > 1.5:
                    quality += 10
                
                # 收回速度（15分）
                if i == 1:
                    quality += 15
                elif i == 2:
                    quality += 10
                elif i == 3:
                    quality += 5
                
                # 扫荡深度适中（10分）
                sweep_depth = (support_low - sweep_bar['low']) / support_low
                if 0.005 <= sweep_depth <= 0.02:
                    quality += 10
                elif sweep_depth < 0.005:
                    quality += 5
                
                return {
                    'detected': True,
                    'quality': quality,
                    'support_level': support_low,
                    'sweep_low': sweep_bar['low'],
                    'recovery_close': recovery_bar['close'],
                    'volume_ratio': vol_ratio,
                    'sweep_depth_pct': sweep_depth * 100
                }
        
        return {'detected': False}
    
    def _identify_order_block_strict(self, data):
        """
        严格的订单块识别
        
        核心逻辑：
        1. 找到强力拉升（BOS - 突破前高至少2%）
        2. 拉升前的最后一根阴线 = 订单块
        3. 当前价格回踩到订单块区域（±3%）
        4. 订单块未被完全跌破
        5. 回踩时成交量萎缩
        """
        if len(data) < 30:
            return {'found': False}
        
        recent_data = data.tail(40)
        current_price = data['close'].iloc[-1]
        
        # 寻找强力拉升
        for i in range(len(recent_data) - 5, 10, -1):
            current_high = recent_data['high'].iloc[i]
            prev_high = recent_data['high'].iloc[:i-2].max()
            
            # 突破幅度 >= 2%
            breakthrough = current_high > prev_high * 1.02
            
            # 持续性：至少2根阳线
            bullish_count = sum(1 for j in range(i, min(i+4, len(recent_data)))
                              if recent_data.iloc[j]['close'] > recent_data.iloc[j]['open'])
            
            if breakthrough and bullish_count >= 2:
                # 找突破前的最后阴线
                for j in range(i-1, max(0, i-6), -1):
                    bar = recent_data.iloc[j]
                    
                    if bar['close'] < bar['open']:  # 阴线
                        ob_high = bar['high']
                        ob_low = bar['low']
                        ob_mid = (ob_high + ob_low) / 2
                        
                        # 当前价格是否回踩到订单块
                        distance = abs(current_price - ob_mid) / ob_mid
                        
                        if distance < 0.03:  # 3%范围内
                            # 检查是否被跌破
                            broken = any(data.iloc[k]['close'] < ob_low * 0.98 
                                       for k in range(max(0, len(data)-10), len(data)))
                            
                            if not broken:
                                # 计算质量分数
                                quality = 0
                                
                                # 订单块成交量（30分）
                                avg_vol = recent_data['volume'].mean()
                                ob_vol_ratio = bar['volume'] / avg_vol if avg_vol > 0 else 0
                                if ob_vol_ratio > 2.0:
                                    quality += 30
                                elif ob_vol_ratio > self.params['order_block_strength']:
                                    quality += 20
                                elif ob_vol_ratio > self.params['volume_surge_ratio']:
                                    quality += 15
                                
                                # 突破强度（25分）
                                displacement = (current_high - prev_high) / prev_high
                                if displacement > 0.05:
                                    quality += 25
                                elif displacement > 0.03:
                                    quality += 20
                                elif displacement > 0.02:
                                    quality += 15
                                
                                # 回踩精度（25分）
                                accuracy = 1 - (distance / 0.03)
                                quality += accuracy * 25
                                
                                # 回踩成交量萎缩（20分）
                                current_vol = data['volume'].tail(5).mean()
                                if current_vol < avg_vol * 0.8:
                                    quality += 20
                                elif current_vol < avg_vol:
                                    quality += 10
                                
                                return {
                                    'found': True,
                                    'quality': quality,
                                    'ob_high': ob_high,
                                    'ob_low': ob_low,
                                    'ob_mid': ob_mid,
                                    'distance_pct': distance * 100,
                                    'breakout_strength_pct': displacement * 100
                                }
        
        return {'found': False}
    
    def _detect_fvg_strict(self, data):
        """
        严格的FVG检测
        
        核心逻辑：
        1. 三根K线形成缺口（bar1高点 < bar3低点）
        2. 缺口大小 >= 0.3%
        3. 缺口未被完全回填（回填 < 50%）
        4. 当前价格在缺口附近（5%范围内）
        5. 缺口形成时成交量放大
        """
        if len(data) < 10:
            return {'detected': False}
        
        current_price = data['close'].iloc[-1]
        
        # 检查最近10根K线
        for i in range(len(data) - 3, max(0, len(data) - 10), -1):
            bar1 = data.iloc[i]
            bar2 = data.iloc[i+1]
            bar3 = data.iloc[i+2]
            
            # 看涨FVG
            if bar1['high'] < bar3['low']:
                gap_low = bar1['high']
                gap_high = bar3['low']
                gap_size = (gap_high - gap_low) / gap_low
                
                # 条件1：缺口大小 >= fvg_min_gap_ratio
                if gap_size < self.params['fvg_min_gap_ratio']:
                    continue
                
                # 条件2：bar2是强力阳线
                bar2_bullish = bar2['close'] > bar2['open']
                bar2_body = abs(bar2['close'] - bar2['open'])
                bar2_range = bar2['high'] - bar2['low']
                strong_bar2 = bar2_body > bar2_range * 0.6
                
                if not (bar2_bullish and strong_bar2):
                    continue
                
                # 条件3：检查回填程度
                filled_pct = 0
                for j in range(i+3, len(data)):
                    if data.iloc[j]['low'] <= gap_low:
                        filled_pct = 1.0
                        break
                    elif data.iloc[j]['low'] < gap_high:
                        filled_pct = (gap_high - data.iloc[j]['low']) / (gap_high - gap_low)
                
                if filled_pct >= 0.5:  # 回填超过50%
                    continue
                
                # 条件4：当前价格位置
                gap_mid = (gap_high + gap_low) / 2
                in_gap = gap_low <= current_price <= gap_high
                distance = abs(current_price - gap_mid) / gap_mid
                
                if distance > 0.05 and not in_gap:  # 超过5%且不在缺口内
                    continue
                
                # 条件5：成交量确认（使用volume_surge_ratio参数）
                avg_vol = data['volume'].tail(20).mean()
                bar2_vol_ratio = bar2['volume'] / avg_vol if avg_vol > 0 else 0
                
                if bar2_vol_ratio < self.params['volume_surge_ratio']:  # 成交量未放大
                    continue
                
                # 计算质量分数
                quality = 0
                
                # 缺口大小（25分）
                if gap_size > 0.01:
                    quality += 25
                elif gap_size > 0.005:
                    quality += 20
                elif gap_size > self.params['fvg_min_gap_ratio']:
                    quality += 15
                
                # 未回填程度（30分）
                if filled_pct == 0:
                    quality += 30
                elif filled_pct < 0.2:
                    quality += 25
                elif filled_pct < 0.5:
                    quality += 20
                
                # 当前价格位置（25分）
                if in_gap:
                    quality += 25
                elif distance < 0.02:
                    quality += 20
                elif distance < 0.05:
                    quality += 15
                
                # 成交量（20分）
                if bar2_vol_ratio > 2.0:
                    quality += 20
                elif bar2_vol_ratio > self.params['order_block_strength']:
                    quality += 15
                elif bar2_vol_ratio > self.params['volume_surge_ratio']:
                    quality += 10
                
                return {
                    'detected': True,
                    'quality': quality,
                    'gap_high': gap_high,
                    'gap_low': gap_low,
                    'gap_size_pct': gap_size * 100,
                    'filled_pct': filled_pct * 100,
                    'in_gap': in_gap,
                    'distance_pct': distance * 100
                }
        
        return {'detected': False}
    
    def _check_entry_timing_strict(self, data, current_price):
        """
        严格的入场时机检查
        
        避免追高的条件：
        1. 当日涨幅 < 3%
        2. 距离近期高点 >= 3%
        3. 没有连续3天大涨
        4. 成交量不是异常放大（< 3倍）
        5. 距离支撑位 < 8%
        """
        if len(data) < 10:
            return {'is_good': False}
        
        recent_data = data.tail(10)
        current_bar = data.iloc[-1]
        prev_close = data.iloc[-2]['close']
        
        # 条件1：当日涨幅
        daily_return = (current_bar['close'] - prev_close) / prev_close
        condition1 = daily_return < 0.03
        
        # 条件2：距离高点
        recent_high = recent_data['high'].max()
        distance_from_high = (recent_high - current_price) / recent_high
        condition2 = distance_from_high >= 0.03
        
        # 条件3：连续大涨检查
        consecutive_rises = 0
        for i in range(len(recent_data) - 3, len(recent_data)):
            if i > 0:
                ret = (recent_data.iloc[i]['close'] - recent_data.iloc[i-1]['close']) / recent_data.iloc[i-1]['close']
                if ret > 0.02:
                    consecutive_rises += 1
                else:
                    break
        condition3 = consecutive_rises < 3
        
        # 条件4：成交量
        current_volume = data['volume'].iloc[-1]
        avg_volume = recent_data['volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        condition4 = volume_ratio < 3.0
        
        # 条件5：距离支撑
        recent_low = recent_data['low'].min()
        distance_from_support = (current_price - recent_low) / recent_low
        condition5 = distance_from_support < 0.08
        
        is_good = condition1 and condition2 and condition3 and condition4 and condition5
        
        return {
            'is_good': is_good,
            'daily_return_pct': daily_return * 100,
            'distance_from_high_pct': distance_from_high * 100,
            'consecutive_rises': consecutive_rises,
            'volume_ratio': volume_ratio,
            'distance_from_support_pct': distance_from_support * 100
        }
    
    def _calculate_confidence_v2(self, market_structure, liquidity_grab, 
                                 order_block, fvg, timing_check, trend_analysis, data):
        """
        置信度计算V2 - 基于信号质量的加权 + 趋势强度 + 趋势线位置
        
        核心改进：
        1. 趋势强度作为核心评分因素（避免反弹误判）
        2. 多周期一致性加成
        3. 趋势线位置优化
        
        目标：让置信度真实反映胜率
        - 60-70分：单一信号，基础质量，趋势一般
        - 70-80分：双重信号或高质量单信号，趋势较强
        - 80-90分：三重共振或双重高质量 + 强趋势 + 好位置
        - 90+分：完美设置（极少）
        """
        confidence = 0
        
        # ========== 第一部分：趋势强度评分（最高25分）==========
        # 这是核心改进，避免在弱趋势或反弹中给高分
        
        trend_strength = market_structure.get('trend_strength', 0)
        
        if trend_strength >= 80:
            confidence += 25  # 极强趋势
        elif trend_strength >= 70:
            confidence += 22  # 强趋势
        elif trend_strength >= 60:
            confidence += 18  # 较强趋势
        elif trend_strength >= 50:
            confidence += 14  # 中等趋势
        elif trend_strength >= 40:
            confidence += 10  # 弱趋势
        else:
            confidence += 5   # 很弱的趋势
        
        # 多周期一致性加成（最高5分）
        consistency_score = market_structure.get('consistency_score', 0)
        if consistency_score == 3:  # 三个周期完全一致
            confidence += 5
        elif consistency_score == 2:
            confidence += 3
        
        # 均线排列加成（最高5分）
        if market_structure.get('ma_alignment', False):
            confidence += 5
        
        # 反弹惩罚（重要！）
        if market_structure.get('is_bounce', False):
            confidence -= 15  # 如果是反弹，大幅降低置信度
        
        # ========== 第二部分：核心信号分数（最高40分）==========
        
        signal_scores = []
        
        if liquidity_grab['detected']:
            signal_scores.append(liquidity_grab['quality'] * 0.20)
        
        if order_block['found']:
            signal_scores.append(order_block['quality'] * 0.20)
        
        if fvg['detected']:
            signal_scores.append(fvg['quality'] * 0.20)
        
        # 取最高的两个信号分数
        signal_scores.sort(reverse=True)
        confidence += sum(signal_scores[:2])
        
        # 多重确认加成（最高10分）
        num_signals = len(signal_scores)
        if num_signals >= 3:
            confidence += 10
        elif num_signals >= 2:
            confidence += 7
        elif num_signals >= 1:
            confidence += 3
        
        # ========== 第三部分：趋势线位置（最高15分）==========
        
        entry_quality = trend_analysis['entry_quality']
        if entry_quality >= 90:
            confidence += 15
        elif entry_quality >= 80:
            confidence += 13
        elif entry_quality >= 70:
            confidence += 11
        elif entry_quality >= 60:
            confidence += 8
        else:
            confidence += 4
        
        # 趋势线位置额外加成（最高5分）
        if trend_analysis['current_position'] == 'near_support':
            confidence += 5
        elif trend_analysis['current_position'] == 'mid_channel':
            confidence += 2
        
        # ========== 第四部分：入场时机（最高10分）==========
        
        if timing_check['is_good']:
            timing_score = 0
            if timing_check['daily_return_pct'] < 1:
                timing_score += 3
            if timing_check['distance_from_high_pct'] > 5:
                timing_score += 3
            if timing_check['volume_ratio'] < 2:
                timing_score += 2
            if timing_check['distance_from_support_pct'] < 5:
                timing_score += 2
            confidence += timing_score
        
        # ========== 第五部分：技术指标辅助（最高10分）==========
        
        indicators = self.tech_indicators.calculate_all_indicators(data)
        if indicators:
            # RSI
            if len(indicators['rsi']) > 0:
                rsi = indicators['rsi'][-1]
                if 40 < rsi < 65:
                    confidence += 5
                elif rsi > 75:
                    confidence -= 5
            
            # MACD
            if len(indicators['macd']) > 0:
                if indicators['macd'][-1] > indicators['macd_signal'][-1]:
                    confidence += 5
        
        return min(confidence, 100)
    
    def _calculate_stop_loss(self, current_price, liquidity_grab, order_block, fvg, atr, trend_analysis):
        """
        计算止损位 - 考虑趋势线支撑和趋势强度
        
        核心改进：
        1. 使用趋势线作为动态止损
        2. 根据趋势强度调整止损缓冲
        3. 趋势越强，止损越宽松（给予更多空间）
        """
        stop_candidates = []
        
        if liquidity_grab['detected']:
            stop_candidates.append(liquidity_grab['support_level'] * 0.99)
        
        if order_block['found']:
            stop_candidates.append(order_block['ob_low'] * 0.99)
        
        if fvg['detected']:
            stop_candidates.append(fvg['gap_low'] * 0.99)
        
        # 趋势线止损（核心优化）
        if trend_analysis['uptrend_line']['valid']:
            # 获取建议的止损缓冲（根据趋势强度动态调整）
            stop_buffer = trend_analysis['suggested_stop_buffer']
            
            # 使用趋势线作为动态止损
            current_idx = len(trend_analysis.get('data_length', 0)) if 'data_length' in trend_analysis else 0
            trend_support = (trend_analysis['uptrend_line']['slope'] * current_idx + 
                           trend_analysis['uptrend_line']['intercept'])
            
            # 根据趋势强度调整止损位
            # 强趋势：趋势线下方5-6%
            # 中等趋势：趋势线下方4-5%
            # 弱趋势：趋势线下方3-4%
            trend_stop = trend_support * (1 - stop_buffer)
            stop_candidates.append(trend_stop)
            
            # 记录趋势强度信息（用于调试）
            trend_strength = trend_analysis['trend_strength']
            # print(f"趋势强度: {trend_strength:.0f}, 止损缓冲: {stop_buffer*100:.1f}%")
        
        # ATR止损
        stop_candidates.append(current_price - atr * 1.5)
        
        # 选择最低的止损位（止损必须低于当前价格）
        stop_loss = min(stop_candidates) if stop_candidates else current_price * 0.97
        
        # 根据趋势强度调整最终止损范围
        if trend_analysis.get('trend_strength', 0) >= 60:
            # 强趋势：允许4-6%的止损空间
            min_stop = current_price * 0.94
            max_stop = current_price * 0.96
        else:
            # 弱趋势：保持2-4%的止损空间
            min_stop = current_price * 0.96
            max_stop = current_price * 0.98
        
        # 确保止损在合理范围内且低于当前价格
        stop_loss = min(stop_loss, max_stop)
        stop_loss = max(stop_loss, min_stop)
        
        # 最终验证：止损必须低于当前价格
        if stop_loss >= current_price:
            stop_loss = current_price * 0.97
        
        return stop_loss
    
    def _calculate_target(self, current_price, stop_loss, atr):
        """计算目标价"""
        risk = current_price - stop_loss
        
        # 目标：至少2.5倍风险
        target1 = current_price + risk * 2.5
        
        # 或3倍ATR
        target2 = current_price + atr * 3
        
        # 或8%收益
        target3 = current_price * 1.08
        
        return max(target1, target2, target3)
    
    def _generate_signal_details(self, market_structure, liquidity_grab, 
                                 order_block, fvg, timing_check, trend_analysis, confidence):
        """生成信号详情 - 包含多周期趋势分析和趋势强度信息"""
        signals = []
        
        # ========== 多周期趋势分析（核心改进）==========
        
        trend_strength = market_structure.get('trend_strength', 0)
        strength_text = market_structure.get('strength', 'unknown')
        
        # 趋势强度评级
        if trend_strength >= 80:
            strength_emoji = "🔥"
            strength_desc = "极强"
        elif trend_strength >= 70:
            strength_emoji = "💪"
            strength_desc = "强"
        elif trend_strength >= 60:
            strength_emoji = "👍"
            strength_desc = "较强"
        elif trend_strength >= 50:
            strength_emoji = "👌"
            strength_desc = "中等"
        elif trend_strength >= 40:
            strength_emoji = "⚠️"
            strength_desc = "弱"
        else:
            strength_emoji = "❌"
            strength_desc = "很弱"
        
        signals.append(f"{strength_emoji} 趋势强度: {strength_desc} ({trend_strength:.0f}/100)")
        
        # 多周期一致性
        consistency = market_structure.get('consistency_score', 0)
        long_up = market_structure.get('long_trend_up', False)
        mid_up = market_structure.get('mid_trend_up', False)
        short_up = market_structure.get('short_trend_up', False)
        
        trend_status = []
        if long_up:
            trend_status.append("长期↑")
        else:
            trend_status.append("长期↓")
        
        if mid_up:
            trend_status.append("中期↑")
        else:
            trend_status.append("中期↓")
        
        if short_up:
            trend_status.append("短期↑")
        else:
            trend_status.append("短期↓")
        
        consistency_text = " | ".join(trend_status)
        signals.append(f"✓ 多周期趋势: {consistency_text} (一致性:{consistency}/3)")
        
        # 均线数据
        ma_long = market_structure.get('ma_long', 0)
        ma_mid = market_structure.get('ma_mid', 0)
        ma_short = market_structure.get('ma_short', 0)
        
        signals.append(f"  MA90:¥{ma_long:.2f} | MA30:¥{ma_mid:.2f} | MA5:¥{ma_short:.2f}")
        
        # 均线排列
        if market_structure.get('ma_alignment', False):
            signals.append(f"  ✓ 多头排列 (MA5 > MA30 > MA90)")
        
        # 反弹警告
        if market_structure.get('is_bounce', False):
            signals.append(f"  ⚠️ 警告: 可能是下降趋势中的反弹")
        
        # 价格相对长期均线位置
        price_vs_long = market_structure.get('price_vs_long_pct', 0)
        signals.append(f"  价格相对MA90: {price_vs_long:+.1f}%")
        
        # ========== 趋势线分析 ==========
        
        if trend_analysis['uptrend_line']['valid']:
            entry_quality = trend_analysis['entry_quality']
            position = trend_analysis['current_position']
            
            position_text = {
                'near_support': '接近支撑线',
                'mid_channel': '通道中部',
                'near_resistance': '接近阻力线',
                'below_support': '跌破支撑',
                'above_channel': '超出通道'
            }.get(position, position)
            
            signals.append(f"✓ 趋势线位置: {position_text} (入场质量:{entry_quality:.0f}/100)")
            
            # 显示止损缓冲
            stop_buffer = trend_analysis['suggested_stop_buffer']
            signals.append(f"  建议止损缓冲: {stop_buffer*100:.1f}%")
            
            if trend_analysis['analysis'].get('distance_from_support_pct') is not None:
                dist = trend_analysis['analysis']['distance_from_support_pct']
                signals.append(f"  距离支撑线: {dist:+.1f}%")
            
            # 显示趋势线持续时间
            if 'start_idx' in trend_analysis['uptrend_line'] and 'end_idx' in trend_analysis['uptrend_line']:
                duration = trend_analysis['uptrend_line']['end_idx'] - trend_analysis['uptrend_line']['start_idx']
                signals.append(f"  趋势持续: {duration}天")
        
        # ========== 核心SMC信号 ==========
        
        if liquidity_grab['detected']:
            signals.append(f"✓ 流动性猎取 (质量:{liquidity_grab['quality']:.0f}/100)")
            signals.append(f"  支撑:¥{liquidity_grab['support_level']:.2f} 扫荡深度:{liquidity_grab['sweep_depth_pct']:.2f}%")
        
        if order_block['found']:
            signals.append(f"✓ 订单块回踩 (质量:{order_block['quality']:.0f}/100)")
            signals.append(f"  区间:¥{order_block['ob_low']:.2f}-¥{order_block['ob_high']:.2f}")
        
        if fvg['detected']:
            signals.append(f"✓ 公允价值缺口 (质量:{fvg['quality']:.0f}/100)")
            signals.append(f"  缺口:¥{fvg['gap_low']:.2f}-¥{fvg['gap_high']:.2f} 回填:{fvg['filled_pct']:.0f}%")
        
        # ========== 入场时机 ==========
        
        if timing_check['is_good']:
            signals.append(f"✓ 入场时机良好 (当日涨幅:{timing_check['daily_return_pct']:.1f}%)")
        else:
            signals.append(f"⚠️ 入场时机一般 (当日涨幅:{timing_check['daily_return_pct']:.1f}%)")
        
        signals.append(f"综合置信度: {confidence:.0f}/100")
        
        return signals
    
    def detect_bearish_signals(self, data):
        """
        检测看空信号
        
        看空信号包括：
        1. 跌破关键支撑位（MA50/MA30）
        2. 看空K线形态（射击之星、乌云盖顶、三只乌鸦等）
        3. 技术指标看空（MACD死叉、RSI超买后回落）
        4. 成交量异常放大的阴线
        
        返回: {'detected': bool, 'confidence': float, 'reasons': list}
        """
        try:
            if len(data) < 30:
                return {'detected': False, 'confidence': 0, 'reasons': []}
            
            reasons = []
            confidence = 0
            
            current_bar = data.iloc[-1]
            current_price = current_bar['close']
            
            # 1. 检查是否跌破关键均线
            ma30 = talib.SMA(data['close'].values, timeperiod=30)
            ma50 = talib.SMA(data['close'].values, timeperiod=50) if len(data) >= 50 else None
            
            if current_price < ma30[-1] * 0.98:  # 跌破MA30超过2%
                reasons.append('跌破MA30')
                confidence += 25
            
            if ma50 is not None and current_price < ma50[-1] * 0.98:  # 跌破MA50超过2%
                reasons.append('跌破MA50')
                confidence += 30
            
            # 2. 检查看空K线形态
            # 射击之星
            body = abs(current_bar['close'] - current_bar['open'])
            upper_shadow = current_bar['high'] - max(current_bar['open'], current_bar['close'])
            lower_shadow = min(current_bar['open'], current_bar['close']) - current_bar['low']
            
            if upper_shadow > body * 2 and lower_shadow < body * 0.3:
                reasons.append('射击之星形态')
                confidence += 20
            
            # 乌云盖顶（两根K线形态）
            if len(data) >= 2:
                prev_bar = data.iloc[-2]
                if (prev_bar['close'] > prev_bar['open'] and  # 前一根是阳线
                    current_bar['close'] < current_bar['open'] and  # 当前是阴线
                    current_bar['open'] > prev_bar['high'] and  # 跳空高开
                    current_bar['close'] < (prev_bar['open'] + prev_bar['close']) / 2):  # 收盘在前一根中点下方
                    reasons.append('乌云盖顶形态')
                    confidence += 25
            
            # 三只乌鸦
            if len(data) >= 3:
                last_three = data.tail(3)
                all_bearish = all(row['close'] < row['open'] for _, row in last_three.iterrows())
                descending = (last_three.iloc[0]['close'] > last_three.iloc[1]['close'] > 
                             last_three.iloc[2]['close'])
                if all_bearish and descending:
                    reasons.append('三只乌鸦形态')
                    confidence += 30
            
            # 3. 技术指标看空
            # MACD死叉
            macd, macd_signal, macd_hist = talib.MACD(data['close'].values)
            if len(macd) >= 2:
                if macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:  # 刚刚死叉
                    reasons.append('MACD死叉')
                    confidence += 20
                elif macd[-1] < macd_signal[-1] and macd_hist[-1] < 0:  # 持续看空
                    reasons.append('MACD看空')
                    confidence += 15
            
            # RSI超买后回落
            rsi = talib.RSI(data['close'].values, timeperiod=14)
            if len(rsi) >= 2:
                if rsi[-2] > 70 and rsi[-1] < 65:  # 从超买区回落
                    reasons.append('RSI超买回落')
                    confidence += 15
            
            # 4. 成交量异常放大的阴线
            avg_volume = data['volume'].tail(20).mean()
            if (current_bar['close'] < current_bar['open'] and  # 阴线
                current_bar['volume'] > avg_volume * 2):  # 成交量放大2倍
                reasons.append('放量下跌')
                confidence += 20
            
            # 判断是否检测到看空信号（置信度>=50即认为有看空信号）
            detected = confidence >= 50
            
            return {
                'detected': detected,
                'confidence': min(confidence, 100),
                'reasons': reasons
            }
            
        except Exception as e:
            print(f"检测看空信号时出错: {e}")
            return {'detected': False, 'confidence': 0, 'reasons': []}
    
    def has_buy_signal(self, stock_code):
        """
        检查指定股票当天是否有买入信号
        
        用于卖出规则：只有当天存在买入信号时，才允许因止损/趋势下破而卖出
        
        返回: bool
        """
        try:
            result = self.screen_stock(stock_code)
            return result is not None and result['signal'] in ['buy', 'strong_buy']
        except:
            return False
    
    def batch_screen(self, stock_list, max_results=50):
        """批量筛选"""
        results = []
        
        print(f"SMC策略V2 - 开始筛选...")
        print(f"候选股票: {len(stock_list)}")
        print("=" * 60)
        
        for idx, stock_code in enumerate(stock_list, 1):
            try:
                result = self.screen_stock(stock_code)
                
                if result and result['signal'] in ['buy', 'strong_buy']:
                    results.append(result)
                    print(f"✓ {stock_code} 置信度:{result['confidence']:.0f}% 信号:{len(result['core_signals'])}")
                
                if idx % 50 == 0:
                    print(f"进度: {idx}/{len(stock_list)} | 找到: {len(results)}")
                
                if len(results) >= max_results:
                    break
                    
            except Exception as e:
                continue
        
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        print("=" * 60)
        print(f"筛选完成！找到 {len(results)} 只股票")
        
        return results

    def generate_report(self, results, filename=None):
        """
        生成详细的筛选报告
        """
        if not results:
            print("没有符合条件的股票")
            return

        print("\n" + "=" * 80)
        print("SMC流动性猎取策略 - 筛选报告")
        print("=" * 80)
        print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"符合条件股票数: {len(results)}")
        print(f"最小置信度要求: {self.params['min_confidence']}%")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n【{i}】 {result['stock_code']}")
            print(f"    信号: {result['signal'].upper()} | 置信度: {result['confidence']:.1f}%")
            print(f"    当前价格: ¥{result['current_price']:.2f}")
            print(f"    建议入场: ¥{result['entry_price']:.2f}")
            print(
                f"    止损位置: ¥{result['stop_loss']:.2f} (风险: {((result['entry_price'] - result['stop_loss']) / result['entry_price'] * 100):.2f}%)")
            print(
                f"    目标价格: ¥{result['target']:.2f} (收益: {((result['target'] - result['entry_price']) / result['entry_price'] * 100):.2f}%)")
            print(f"    风险收益比: 1:{result['risk_reward_ratio']:.2f}")
            if 'atr' in result:
                print(f"    ATR波动: ¥{result['atr']:.2f}")
            print(f"\n    信号详情:")
            for signal in result['signals']:
                print(f"    {signal}")
            print("-" * 80)

        # 保存到CSV
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"smc_liquidity_screening_{timestamp}.csv"

        export_data = []
        for result in results:
            export_data.append({
                '股票代码': result['stock_code'],
                '信号': result['signal'],
                '置信度': f"{result['confidence']:.1f}%",
                '当前价格': result['current_price'],
                '入场价': result['entry_price'],
                '止损价': result['stop_loss'],
                '目标价': result['target'],
                '风险%': f"{((result['entry_price'] - result['stop_loss']) / result['entry_price'] * 100):.2f}%",
                '收益%': f"{((result['target'] - result['entry_price']) / result['entry_price'] * 100):.2f}%",
                '风险收益比': f"1:{result['risk_reward_ratio']:.2f}",
                'ATR': result.get('atr', 0),
                '信号数量': len(result['signals'])
            })

        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')

        print(f"\n✓ 报告已保存到: {filename}")
        print("=" * 80)
