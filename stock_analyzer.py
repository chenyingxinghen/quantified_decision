#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票分析脚本 - 基于SMC和Wyckoff两个策略
分析指定股票并给出买入卖出建议
"""

import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import DataFetcher
from smc_liquidity_strategy import SMCLiquidityStrategy
from wyckoff_strategy import WyckoffStrategy
import sys


class StockAnalyzer:
    """股票分析器 - 整合两个策略的分析结果"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.smc_strategy = SMCLiquidityStrategy()
        self.wyckoff_strategy = WyckoffStrategy()
    
    def analyze_stock(self, stock_code):
        """
        分析指定股票
        
        Args:
            stock_code: 股票代码（如 '600519'）
        
        Returns:
            dict: 包含两个策略的分析结果和综合建议
        """
        print("=" * 80)
        print(f"正在分析股票: {stock_code}")
        print("=" * 80)
        
        # 1. 获取最新K线数据
        print("\n【步骤1】获取最新K线数据...")
        try:
            # 先从数据库获取数据
            data = self.data_fetcher.get_stock_data(stock_code, days=300)
            
            # 检查数据是否存在且是否最新
            need_update = False
            if data.empty or len(data) < 60:
                print(f"  数据库中无数据，需要下载...")
                need_update = True
            else:
                last_date = data['date'].iloc[-1]
                from datetime import datetime, timedelta
                now = datetime.now()
                today = now.strftime('%Y-%m-%d')
                today_15pm = now.replace(hour=15, minute=0, second=0, microsecond=0)
                
                # 检查数据是否需要更新
                # 1. 如果最后日期早于今天，肯定需要更新
                # 2. 如果最后日期是今天，且当前时间已过15:00，检查更新时间
                if last_date < today:
                    print(f"  数据库数据过期（最后更新: {last_date}），正在更新...")
                    need_update = True
                elif last_date == today and now >= today_15pm:
                    # 今天且已过15:00，检查数据的更新时间
                    cursor = self.data_fetcher.conn.cursor()
                    cursor.execute('''
                        SELECT date FROM daily_data 
                        WHERE code = ?
                    ''', (stock_code,))
                    result = cursor.fetchone()
                    
                    if result and result[0]:
                        update_time = datetime.fromisoformat(result[0])
                        # 如果更新时间早于今天15:00，需要更新
                        if update_time < today_15pm:
                            print(f"  数据库数据未在15:00后更新（更新时间: {update_time.strftime('%H:%M:%S')}），正在更新...")
                            need_update = True
                        else:
                            print(f"  数据库数据已是最新（最后更新: {last_date} {update_time.strftime('%H:%M:%S')}）")
                    else:
                        # 没有更新时间记录，需要更新
                        print(f"  数据库数据缺少更新时间记录，正在更新...")
                        need_update = True
                else:
                    print(f"  数据库数据已是最新（最后更新: {last_date}）")
            
            # 如果需要更新，则更新数据
            if need_update:
                self.data_fetcher.update_daily_data(stock_code, incremental=True)
                # 重新获取数据
                data = self.data_fetcher.get_stock_data(stock_code, days=300)
                if data.empty or len(data) < 60:
                    print(f"❌ 错误: 股票 {stock_code} 数据不足，无法分析")
                    return None
                
                print(f"  ✓ 数据更新完成")
            
            print(f"✓ 成功获取 {len(data)} 天的K线数据")
            print(f"  数据范围: {data['date'].iloc[0]} 至 {data['date'].iloc[-1]}")
            
            # 显示最新K线信息
            latest = data.iloc[-1]
            print(f"\n  最新K线 ({latest['date']}):")
            print(f"    开盘: ¥{latest['open']:.2f}")
            print(f"    最高: ¥{latest['high']:.2f}")
            print(f"    最低: ¥{latest['low']:.2f}")
            print(f"    收盘: ¥{latest['close']:.2f}")
            print(f"    成交量: {latest['volume']:,.0f}")
            
            # 计算涨跌幅
            if len(data) >= 2:
                prev_close = data.iloc[-2]['close']
                change_pct = (latest['close'] - prev_close) / prev_close * 100
                print(f"    涨跌幅: {change_pct:+.2f}%")
            
        except Exception as e:
            print(f"❌ 获取数据失败: {e}")
            return None
        
        # 2. SMC策略分析
        print("\n【步骤2】SMC流动性猎取策略分析...")
        print("-" * 80)
        try:
            smc_result = self.smc_strategy.screen_stock(stock_code)
            
            if smc_result:
                self._print_smc_analysis(smc_result)
            else:
                # 给出详细的不符合原因分析
                self._print_smc_rejection_analysis(stock_code, data)
        except Exception as e:
            print(f"❌ SMC策略分析失败: {e}")
            import traceback
            traceback.print_exc()
            smc_result = None
        
        # 3. Wyckoff策略分析
        print("\n【步骤3】Wyckoff Spring策略分析...")
        print("-" * 80)
        try:
            wyckoff_result = self.wyckoff_strategy.wyckoff_accumulation_strategy(stock_code)
            
            if wyckoff_result:
                self._print_wyckoff_analysis(wyckoff_result)
            else:
                # 给出详细的不符合原因分析
                self._print_wyckoff_rejection_analysis(stock_code, data)
        except Exception as e:
            print(f"❌ Wyckoff策略分析失败: {e}")
            import traceback
            traceback.print_exc()
            wyckoff_result = None
        
        # 4. 综合建议
        print("\n【步骤4】综合建议")
        print("=" * 80)
        
        recommendation = self._generate_recommendation(smc_result, wyckoff_result, data)
        
        self._print_recommendation(recommendation)
        
        # 5. 检测看空信号
        print("\n【步骤5】风险提示")
        print("-" * 80)
        bearish_signals = self.smc_strategy.detect_bearish_signals(data)
        
        if bearish_signals['detected']:
            print(f"⚠️ 检测到看空信号 (置信度: {bearish_signals['confidence']:.0f}%)")
            print("   看空原因:")
            for reason in bearish_signals['reasons']:
                print(f"   - {reason}")
            print("\n   建议: 谨慎操作，考虑等待或减仓")
        else:
            print("✓ 未检测到明显看空信号")
        
        print("=" * 80)
        
        return {
            'stock_code': stock_code,
            'data': data,
            'smc_result': smc_result,
            'wyckoff_result': wyckoff_result,
            'recommendation': recommendation,
            'bearish_signals': bearish_signals
        }
    
    def _print_smc_analysis(self, result):
        """打印SMC策略分析结果"""
        print(f"✓ SMC策略信号: {result['signal'].upper()}")
        print(f"  置信度: {result['confidence']:.0f}%")
        print(f"  当前价格: ¥{result['current_price']:.2f}")
        print(f"  建议入场: ¥{result['entry_price']:.2f}")
        print(f"  止损位置: ¥{result['stop_loss']:.2f} (风险: {((result['entry_price'] - result['stop_loss']) / result['entry_price'] * 100):.2f}%)")
        print(f"  目标价格: ¥{result['target']:.2f} (收益: {((result['target'] - result['entry_price']) / result['entry_price'] * 100):.2f}%)")
        print(f"  风险收益比: 1:{result['risk_reward_ratio']:.2f}")
        
        print(f"\n  核心信号: {', '.join(result['core_signals'])}")
        print(f"\n  详细分析:")
        for signal in result['signals']:
            print(f"    {signal}")
    
    def _print_wyckoff_analysis(self, result):
        """打印Wyckoff策略分析结果"""
        print(f"✓ Wyckoff策略信号: {result['signal'].upper()}")
        print(f"  置信度: {result['confidence']:.0f}%")
        print(f"  当前价格: ¥{result['current_price']:.2f}")
        print(f"  建议入场: ¥{result['entry_price']:.2f}")
        print(f"  止损位置: ¥{result['stop_loss']:.2f} (风险: {((result['entry_price'] - result['stop_loss']) / result['entry_price'] * 100):.2f}%)")
        print(f"  目标价格: ¥{result['target']:.2f} (收益: {((result['target'] - result['entry_price']) / result['entry_price'] * 100):.2f}%)")
        print(f"  风险收益比: 1:{result['risk_reward_ratio']:.2f}")
        
        print(f"\n  满足条件:")
        for condition in result['conditions_met']:
            print(f"    ✓ {condition}")
    
    def _generate_recommendation(self, smc_result, wyckoff_result, data):
        """生成综合建议"""
        current_price = data['close'].iloc[-1]
        
        # 统计信号
        signals = []
        if smc_result and smc_result['signal'] in ['buy', 'strong_buy']:
            signals.append('smc')
        if wyckoff_result and wyckoff_result['signal'] in ['buy', 'strong_buy']:
            signals.append('wyckoff')
        
        # 计算综合置信度
        confidences = []
        if smc_result:
            confidences.append(smc_result['confidence'])
        if wyckoff_result:
            confidences.append(wyckoff_result['confidence'])
        
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # 计算综合止损和目标
        stop_losses = []
        targets = []
        
        if smc_result:
            stop_losses.append(smc_result['stop_loss'])
            targets.append(smc_result['target'])
        
        if wyckoff_result:
            stop_losses.append(wyckoff_result['stop_loss'])
            targets.append(wyckoff_result['target'])
        
        # 使用最高的止损（更保守）和最低的目标（更保守）
        recommended_stop = max(stop_losses) if stop_losses else current_price * 0.95
        recommended_target = min(targets) if targets else current_price * 1.10
        
        # 生成操作建议
        if len(signals) == 2:
            # 两个策略都给出买入信号
            action = "强烈建议买入"
            reason = "两个策略同时给出买入信号，信号强度高"
            strength = "强"
        elif len(signals) == 1:
            # 只有一个策略给出买入信号
            if avg_confidence >= 80:
                action = "建议买入"
                reason = f"{signals[0].upper()}策略给出买入信号，置信度较高"
                strength = "中"
            else:
                action = "可以考虑买入"
                reason = f"{signals[0].upper()}策略给出买入信号，但置信度一般"
                strength = "弱"
        else:
            # 没有买入信号
            action = "观望"
            reason = "两个策略均未给出明确买入信号"
            strength = "无"
        
        return {
            'action': action,
            'reason': reason,
            'strength': strength,
            'signals': signals,
            'avg_confidence': avg_confidence,
            'entry_price': current_price,
            'stop_loss': recommended_stop,
            'target': recommended_target,
            'risk_pct': (current_price - recommended_stop) / current_price * 100,
            'reward_pct': (recommended_target - current_price) / current_price * 100
        }
    
    def _print_recommendation(self, rec):
        """打印综合建议"""
        print(f"操作建议: {rec['action']}")
        print(f"信号强度: {rec['strength']}")
        print(f"理由: {rec['reason']}")
        
        if rec['signals']:
            print(f"\n给出信号的策略: {', '.join([s.upper() for s in rec['signals']])}")
            print(f"综合置信度: {rec['avg_confidence']:.0f}%")
        
        if rec['action'] != "观望":
            print(f"\n建议操作:")
            print(f"  入场价格: ¥{rec['entry_price']:.2f}")
            print(f"  止损位置: ¥{rec['stop_loss']:.2f} (风险: {rec['risk_pct']:.2f}%)")
            print(f"  目标价格: ¥{rec['target']:.2f} (收益: {rec['reward_pct']:.2f}%)")
            print(f"  风险收益比: 1:{rec['reward_pct']/rec['risk_pct']:.2f}")
            
            print(f"\n执行规则:")
            print(f"  1. 买入: 在 ¥{rec['entry_price']:.2f} 附近分批买入")
            print(f"  2. 止损: 跌破 ¥{rec['stop_loss']:.2f} 立即止损")
            print(f"  3. 止盈: 达到 ¥{rec['target']:.2f} 分批止盈")
            print(f"  4. 仓位: 建议不超过总资金的10-20%")
    
    def _print_smc_rejection_analysis(self, stock_code: str, data: pd.DataFrame):
        """打印SMC策略不符合的详细分析"""
        print("✗ SMC策略: 不符合买入条件")
        print("\n  详细分析:")
        
        # 1. 市场结构分析
        market_structure = self.smc_strategy._check_market_structure_strict(data)
        print(f"\n  【市场结构】")
        print(f"    趋势方向: {'上升趋势 ✓' if market_structure['is_uptrend'] else '非上升趋势 ✗'}")
        if not market_structure['is_uptrend']:
            print(f"    原因: {market_structure.get('reason', '未检测到明确的上升趋势')}")
        
        # 2. 流动性猎取分析
        liquidity_grab = self.smc_strategy._detect_liquidity_grab_strict(data)
        print(f"\n  【流动性猎取】")
        print(f"    检测到流动性猎取: {'是 ✓' if liquidity_grab['detected'] else '否 ✗'}")
        if liquidity_grab['detected']:
            print(f"    猎取类型: {liquidity_grab['type']}")
            print(f"    猎取日期: {liquidity_grab['date']}")
            print(f"    猎取价格: ¥{liquidity_grab['liquidity_price']:.2f}")
            print(f"    反弹收盘: ¥{liquidity_grab['bounce_close']:.2f}")
        else:
            print(f"    说明: 未检测到明显的流动性猎取形态")
        
        # 3. 订单块分析
        order_block = self.smc_strategy._identify_order_block_strict(data)
        print(f"\n  【订单块】")
        print(f"    检测到订单块: {'是 ✓' if order_block['found'] else '否 ✗'}")
        if order_block['found']:
            print(f"    价格范围: ¥{order_block['ob_low']:.2f} - ¥{order_block['ob_high']:.2f}")
            print(f"    质量评分: {order_block['quality']:.1f}/100")
            print(f"    回踩精度: {order_block['distance_pct']:.2f}%")
            print(f"    突破强度: {order_block['breakout_strength_pct']:.2f}%")
        else:
            print(f"    状态: 未检测到有效的订单块")
        
        # 4. 公允价值缺口分析
        fvg = self.smc_strategy._detect_fvg_strict(data)
        print(f"\n  【公允价值缺口 (FVG)】")
        print(f"    检测到FVG: {'是 ✓' if fvg['detected'] else '否 ✗'}")
        if fvg['detected']:
            print(f"    缺口范围: ¥{fvg['gap_low']:.2f} - ¥{fvg['gap_high']:.2f}")
            print(f"    质量评分: {fvg['quality']:.1f}/100")
        else:
            print(f"    状态: 未检测到FVG")
        
        # 5. 时机检查
        current_price = data['close'].iloc[-1]
        timing_check = self.smc_strategy._check_entry_timing_strict(data, current_price)
        print(f"\n  【入场时机】")
        print(f"    时机评估: {'良好 ✓' if timing_check['is_good'] else '不佳 ✗'}")
        print(f"    当日涨幅: {timing_check['daily_return_pct']:.2f}%")
        print(f"    距离高点: {timing_check['distance_from_high_pct']:.2f}%")
        print(f"    连续上涨天数: {timing_check['consecutive_rises']}天")
        print(f"    成交量比率: {timing_check['volume_ratio']:.2f}x")
        print(f"    距离支撑: {timing_check['distance_from_support_pct']:.2f}%")
        
        # 6. 看空信号检测
        bearish_signals = self.smc_strategy.detect_bearish_signals(data)
        print(f"\n  【看空信号检测】")
        if bearish_signals['detected']:
            print(f"    ⚠️ 检测到看空信号 (置信度: {bearish_signals['confidence']:.0f}%)")
            print(f"    看空原因:")
            for reason in bearish_signals['reasons']:
                print(f"      - {reason}")
        else:
            print(f"    状态: 未检测到看空信号 ✓")
        
        # 7. 综合评估
        print(f"\n  【综合评估】")
        issues = []
        if not market_structure['is_uptrend']:
            issues.append("市场结构不符合上升趋势")
        if not liquidity_grab['detected']:
            issues.append("未检测到流动性猎取形态")
        if not order_block['found']:
            issues.append("缺少有效的订单块支撑")
        if not timing_check['is_good']:
            issues.append("入场时机不佳")
        if bearish_signals['detected']:
            issues.append(f"存在看空信号 (置信度{bearish_signals['confidence']:.0f}%)")
        
        if issues:
            print(f"    不符合原因:")
            for issue in issues:
                print(f"      • {issue}")
        else:
            print(f"    说明: 虽然各项指标基本符合，但综合置信度不足")
    
    def _print_wyckoff_rejection_analysis(self, stock_code: str, data: pd.DataFrame):
        """打印Wyckoff策略不符合的详细分析"""
        print("✗ Wyckoff策略: 不符合买入条件")
        print("\n  详细分析:")
        
        # 1. 趋势分析
        from trend_line_analyzer import TrendLineAnalyzer
        trend_analyzer = TrendLineAnalyzer()
        trend_analysis = trend_analyzer.analyze(data)
        
        print(f"\n  【趋势分析】")
        print(f"    当前价格位置: {trend_analysis['current_position']}")
        print(f"    趋势强度: {trend_analysis['trend_strength']:.1f}/100")
        print(f"    入场质量评分: {trend_analysis['entry_quality']:.1f}/100")
        
        # 支撑趋势线
        support_line = trend_analysis['uptrend_line']
        print(f"    支撑趋势线: {'存在 ✓' if support_line['valid'] else '不存在 ✗'}")
        if support_line['valid']:
            print(f"      斜率: {support_line['slope']:.4f}")
            print(f"      触点数量: {support_line['touches']}")
            print(f"      角度: {support_line['angle_degrees']:.1f}°")
            
            # 计算当前支撑价格
            current_idx = len(data) - 1
            current_support = support_line['slope'] * current_idx + support_line['intercept']
            print(f"      当前支撑价: ¥{current_support:.2f}")
        
        # 阻力趋势线
        resistance_line = trend_analysis['downtrend_line']
        print(f"    阻力趋势线: {'存在 ✓' if resistance_line['valid'] else '不存在 ✗'}")
        if resistance_line['valid']:
            print(f"      斜率: {resistance_line['slope']:.4f}")
            print(f"      触点数量: {resistance_line['touches']}")
            print(f"      角度: {resistance_line['angle_degrees']:.1f}°")
        
        # 支撑是否被跌破
        print(f"    支撑跌破: {'是 ✗' if trend_analysis['broken_support'] else '否 ✓'}")
        
        # 2. Spring形态检测
        spring_detected = False
        spring_info = None
        
        # 检测Spring形态的关键特征
        if len(data) >= 20 and support_line['valid']:
            recent_data = data.tail(20)
            current_idx = len(data) - 1
            support_price = support_line['slope'] * current_idx + support_line['intercept']
            
            # 检查最近是否有假跌破后快速反弹
            for i in range(len(recent_data) - 1):
                row = recent_data.iloc[i]
                next_row = recent_data.iloc[i + 1]
                
                # 跌破支撑
                if row['low'] < support_price * 0.98:
                    # 快速反弹
                    if next_row['close'] > support_price:
                        spring_detected = True
                        spring_info = {
                            'date': row['date'],
                            'low': row['low'],
                            'support': support_price,
                            'bounce_close': next_row['close']
                        }
                        break
        
        print(f"\n  【Spring形态】")
        if spring_detected and spring_info:
            print(f"    检测到Spring: ✓")
            print(f"    发生日期: {spring_info['date']}")
            print(f"    最低价: ¥{spring_info['low']:.2f}")
            print(f"    支撑位: ¥{spring_info['support']:.2f}")
            print(f"    反弹收盘: ¥{spring_info['bounce_close']:.2f}")
        else:
            print(f"    检测到Spring: ✗")
            if not support_line['valid']:
                print(f"    说明: 缺少有效的支撑趋势线")
            else:
                print(f"    说明: 未检测到明显的假跌破后快速反弹形态")
        
        # 3. 成交量分析
        print(f"\n  【成交量分析】")
        if len(data) >= 20:
            recent_volume = data['volume'].tail(20)
            avg_volume = recent_volume.mean()
            latest_volume = data['volume'].iloc[-1]
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0
            
            print(f"    最新成交量: {latest_volume:,.0f}")
            print(f"    20日平均: {avg_volume:,.0f}")
            print(f"    成交量比率: {volume_ratio:.2f}x")
            
            if volume_ratio > 1.5:
                print(f"    状态: 成交量放大 ✓")
            elif volume_ratio > 1.2:
                print(f"    状态: 成交量略有放大")
            else:
                print(f"    状态: 成交量未明显放大 ✗")
        
        # 4. 价格位置分析
        print(f"\n  【价格位置】")
        latest_close = data['close'].iloc[-1]
        recent_high = data['high'].tail(60).max()
        recent_low = data['low'].tail(60).min()
        price_range = recent_high - recent_low
        price_position = (latest_close - recent_low) / price_range if price_range > 0 else 0.5
        
        print(f"    当前价格: ¥{latest_close:.2f}")
        print(f"    60日最高: ¥{recent_high:.2f}")
        print(f"    60日最低: ¥{recent_low:.2f}")
        print(f"    价格位置: {price_position*100:.1f}% (0%=最低, 100%=最高)")
        
        if price_position < 0.3:
            print(f"    状态: 处于低位区域 ✓")
        elif price_position < 0.5:
            print(f"    状态: 处于中低位区域")
        else:
            print(f"    状态: 价格位置偏高 ✗")
        
        # 5. 看空信号检测
        bearish_signals = self.smc_strategy.detect_bearish_signals(data)
        print(f"\n  【看空信号检测】")
        if bearish_signals['detected']:
            print(f"    ⚠️ 检测到看空信号 (置信度: {bearish_signals['confidence']:.0f}%)")
            print(f"    看空原因:")
            for reason in bearish_signals['reasons']:
                print(f"      - {reason}")
        else:
            print(f"    状态: 未检测到看空信号 ✓")
        
        # 6. 综合评估
        print(f"\n  【综合评估】")
        issues = []
        if not support_line['valid']:
            issues.append("缺少有效的支撑趋势线")
        if trend_analysis['broken_support']:
            issues.append("支撑趋势线已被跌破")
        if not spring_detected:
            issues.append("未检测到Spring形态")
        if trend_analysis['trend_strength'] < 50:
            issues.append(f"趋势强度不足 ({trend_analysis['trend_strength']:.1f}/100)")
        if trend_analysis['entry_quality'] < 50:
            issues.append(f"入场质量不佳 ({trend_analysis['entry_quality']:.1f}/100)")
        if bearish_signals['detected']:
            issues.append(f"存在看空信号 (置信度{bearish_signals['confidence']:.0f}%)")
        
        if issues:
            print(f"    不符合原因:")
            for issue in issues:
                print(f"      • {issue}")
        else:
            print(f"    说明: 虽然各项指标基本符合，但综合置信度不足")
    
    def batch_analyze(self, stock_codes):
        """批量分析多只股票"""
        results = []
        
        print(f"\n开始批量分析 {len(stock_codes)} 只股票...")
        print("=" * 80)
        
        for i, code in enumerate(stock_codes, 1):
            print(f"\n[{i}/{len(stock_codes)}] 分析 {code}...")
            
            try:
                result = self.analyze_stock(code)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ 分析 {code} 失败: {e}")
            
            print("\n" + "=" * 80)
        
        # 生成汇总报告
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results):
        """打印批量分析汇总"""
        print("\n" + "=" * 80)
        print("批量分析汇总")
        print("=" * 80)
        
        if not results:
            print("没有成功分析的股票")
            return
        
        # 按建议强度分类
        strong_buy = []
        buy = []
        watch = []
        
        for r in results:
            rec = r['recommendation']
            if rec['action'] == "强烈建议买入":
                strong_buy.append(r)
            elif rec['action'] in ["建议买入", "可以考虑买入"]:
                buy.append(r)
            else:
                watch.append(r)
        
        print(f"\n总计分析: {len(results)} 只股票")
        print(f"  强烈建议买入: {len(strong_buy)} 只")
        print(f"  建议买入: {len(buy)} 只")
        print(f"  观望: {len(watch)} 只")
        
        if strong_buy:
            print(f"\n【强烈建议买入】")
            for r in strong_buy:
                rec = r['recommendation']
                print(f"  {r['stock_code']}: 置信度 {rec['avg_confidence']:.0f}%, "
                      f"风险 {rec['risk_pct']:.1f}%, 收益 {rec['reward_pct']:.1f}%")
        
        if buy:
            print(f"\n【建议买入】")
            for r in buy:
                rec = r['recommendation']
                print(f"  {r['stock_code']}: 置信度 {rec['avg_confidence']:.0f}%, "
                      f"风险 {rec['risk_pct']:.1f}%, 收益 {rec['reward_pct']:.1f}%")
        
        print("=" * 80)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python stock_analyzer.py <股票代码1> [股票代码2] ...")
        print("示例: python stock_analyzer.py 600519")
        print("示例: python stock_analyzer.py 600519 000858 601318")
        sys.exit(1)
    
    stock_codes = sys.argv[1:]
    
    analyzer = StockAnalyzer()
    
    if len(stock_codes) == 1:
        # 单只股票分析
        analyzer.analyze_stock(stock_codes[0])
    else:
        # 批量分析
        analyzer.batch_analyze(stock_codes)


if __name__ == "__main__":
    main()
