# 智能参数调优器 - 从历史回测报告中学习最优参数
# 核心思路：分析历史交易，找出高胜率参数组合的共同特征

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import os
from datetime import datetime


class SmartParameterTuner:
    """
    智能参数调优器
    
    核心功能：
    1. 分析历史交易记录，找出盈利交易的共同特征
    2. 统计不同置信度区间的实际胜率
    3. 识别导致亏损的参数设置
    4. 自动调整参数以提高胜率
    """
    
    def __init__(self):
        self.trade_analysis = []
        self.confidence_calibration = {}
        self.parameter_impact = {}
    
    def analyze_historical_trades(self, trades_csv_path: str) -> Dict:
        """
        分析历史交易记录
        
        参数:
            trades_csv_path: 交易记录CSV文件路径
        
        返回:
            分析结果字典
        """
        print("=" * 80)
        print("分析历史交易记录")
        print("=" * 80)
        
        # 读取交易记录
        df = pd.read_csv(trades_csv_path, encoding='utf-8-sig')
        
        if len(df) == 0:
            print("警告: 交易记录为空")
            return {}
        
        # 基础统计
        total_trades = len(df)
        winning_trades = df[df['收益率'].str.rstrip('%').astype(float) > 0]
        losing_trades = df[df['收益率'].str.rstrip('%').astype(float) <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        print(f"总交易数: {total_trades}")
        print(f"盈利交易: {len(winning_trades)} ({win_rate*100:.2f}%)")
        print(f"亏损交易: {len(losing_trades)}")
        print("-" * 80)
        
        # 分析卖出原因
        exit_reasons = df['卖出原因'].value_counts()
        print("\n卖出原因分布:")
        for reason, count in exit_reasons.items():
            pct = count / total_trades * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # 分析持仓时长
        df['买入日期'] = pd.to_datetime(df['买入日期'])
        df['卖出日期'] = pd.to_datetime(df['卖出日期'])
        df['持仓天数'] = (df['卖出日期'] - df['买入日期']).dt.days
        
        print(f"\n持仓时长统计:")
        print(f"  平均: {df['持仓天数'].mean():.1f} 天")
        print(f"  中位数: {df['持仓天数'].median():.1f} 天")
        print(f"  最长: {df['持仓天数'].max()} 天")
        print(f"  最短: {df['持仓天数'].min()} 天")
        
        # 分析收益分布
        df['收益率_数值'] = df['收益率'].str.rstrip('%').astype(float) / 100
        
        print(f"\n收益率统计:")
        print(f"  平均: {df['收益率_数值'].mean()*100:.2f}%")
        print(f"  中位数: {df['收益率_数值'].median()*100:.2f}%")
        print(f"  最大盈利: {df['收益率_数值'].max()*100:.2f}%")
        print(f"  最大亏损: {df['收益率_数值'].min()*100:.2f}%")
        
        # 识别问题模式
        print("\n" + "=" * 80)
        print("问题模式识别")
        print("=" * 80)
        
        # 1. 止损过于频繁
        stop_loss_trades = df[df['卖出原因'] == 'stop_loss']
        if len(stop_loss_trades) > total_trades * 0.3:
            print("⚠️ 问题1: 止损过于频繁")
            print(f"   止损交易占比: {len(stop_loss_trades)/total_trades*100:.1f}%")
            print(f"   建议: 放宽止损距离（增加ATR_STOP_MULTIPLIER）")
        
        # 2. 持仓时间过长
        long_hold_trades = df[df['持仓天数'] > 10]
        if len(long_hold_trades) > total_trades * 0.2:
            print("\n⚠️ 问题2: 持仓时间过长")
            print(f"   超过10天的交易: {len(long_hold_trades)} ({len(long_hold_trades)/total_trades*100:.1f}%)")
            print(f"   建议: 添加时间止损规则")
        
        # 3. 小亏损累积
        small_loss_trades = df[(df['收益率_数值'] < 0) & (df['收益率_数值'] > -0.03)]
        if len(small_loss_trades) > len(losing_trades) * 0.5:
            print("\n⚠️ 问题3: 小亏损累积")
            print(f"   小亏损(<3%)交易: {len(small_loss_trades)} ({len(small_loss_trades)/total_trades*100:.1f}%)")
            print(f"   建议: 提高入场质量要求（增加MIN_CONFIDENCE）")
        
        # 4. 趋势反转导致亏损
        trend_reversal_trades = df[df['卖出原因'].str.contains('trend_reversal|broken_trendline', na=False)]
        if len(trend_reversal_trades) > 0:
            trend_reversal_loss = trend_reversal_trades[trend_reversal_trades['收益率_数值'] < 0]
            if len(trend_reversal_loss) > len(trend_reversal_trades) * 0.6:
                print("\n⚠️ 问题4: 趋势反转识别不及时")
                print(f"   趋势反转亏损率: {len(trend_reversal_loss)/len(trend_reversal_trades)*100:.1f}%")
                print(f"   建议: 提高趋势强度要求（增加MIN_TREND_STRENGTH）")
        
        print("\n" + "=" * 80)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': df['收益率_数值'].mean(),
            'exit_reasons': exit_reasons.to_dict(),
            'avg_holding_days': df['持仓天数'].mean()
        }
    
    def calibrate_confidence_from_trades(self, trades_csv_path: str, 
                                        confidence_column: str = None) -> Dict:
        """
        从交易记录中校准置信度
        
        注意: 需要交易记录中包含置信度信息
        如果CSV中没有置信度列，此功能无法使用
        
        返回:
            {confidence_range: actual_win_rate}
        """
        print("\n" + "=" * 80)
        print("置信度校准")
        print("=" * 80)
        
        df = pd.read_csv(trades_csv_path, encoding='utf-8-sig')
        
        # 检查是否有置信度列
        if confidence_column and confidence_column not in df.columns:
            print(f"警告: CSV中没有找到置信度列 '{confidence_column}'")
            print("可用列:", df.columns.tolist())
            return {}
        
        if confidence_column is None:
            # 尝试自动查找置信度列
            possible_names = ['置信度', 'confidence', 'Confidence', '信号置信度']
            for name in possible_names:
                if name in df.columns:
                    confidence_column = name
                    break
        
        if confidence_column is None:
            print("警告: 无法找到置信度列，跳过校准")
            print("提示: 需要在回测时记录每笔交易的置信度")
            return {}
        
        # 提取置信度和收益率
        df['收益率_数值'] = df['收益率'].str.rstrip('%').astype(float) / 100
        df['置信度_数值'] = df[confidence_column].str.rstrip('%').astype(float) if df[confidence_column].dtype == 'object' else df[confidence_column]
        
        # 按置信度分段统计
        bins = [0, 60, 70, 80, 90, 100]
        df['置信度区间'] = pd.cut(df['置信度_数值'], bins=bins, labels=['60-70', '70-80', '80-90', '90-100'])
        
        calibration_map = {}
        
        print("\n置信度区间 vs 实际胜率:")
        print("-" * 80)
        
        for interval in ['60-70', '70-80', '80-90', '90-100']:
            interval_trades = df[df['置信度区间'] == interval]
            
            if len(interval_trades) > 0:
                actual_win_rate = (interval_trades['收益率_数值'] > 0).sum() / len(interval_trades)
                avg_return = interval_trades['收益率_数值'].mean()
                
                calibration_map[interval] = {
                    'actual_win_rate': actual_win_rate,
                    'avg_return': avg_return,
                    'sample_count': len(interval_trades)
                }
                
                print(f"{interval}%: 样本{len(interval_trades)}笔, "
                      f"实际胜率{actual_win_rate*100:.1f}%, "
                      f"平均收益{avg_return*100:.2f}%")
        
        print("-" * 80)
        
        # 保存校准结果
        self.confidence_calibration = calibration_map
        
        return calibration_map
    
    def suggest_parameter_adjustments(self, analysis_result: Dict) -> Dict:
        """
        基于分析结果建议参数调整
        
        返回:
            {parameter_name: suggested_value}
        """
        print("\n" + "=" * 80)
        print("参数调整建议")
        print("=" * 80)
        
        suggestions = {}
        
        # 规则1: 胜率过低 -> 提高入场标准
        if analysis_result.get('win_rate', 0) < 0.5:
            print("\n📊 胜率偏低 (<50%)")
            suggestions['MIN_CONFIDENCE'] = 85  # 提高最低置信度
            suggestions['MIN_ENTRY_QUALITY'] = 90  # 提高入场质量
            suggestions['MIN_TREND_STRENGTH'] = 85  # 提高趋势强度
            print("   建议: 提高入场标准")
            print(f"   - MIN_CONFIDENCE: 85")
            print(f"   - MIN_ENTRY_QUALITY: 90")
            print(f"   - MIN_TREND_STRENGTH: 85")
        
        # 规则2: 胜率较高但收益低 -> 扩大盈利空间
        elif analysis_result.get('win_rate', 0) > 0.6 and analysis_result.get('avg_return', 0) < 0.03:
            print("\n📊 胜率较高但收益偏低")
            suggestions['ATR_TARGET_MULTIPLIER'] = 4.0  # 提高目标倍数
            suggestions['MIN_RISK_REWARD_RATIO'] = 2.5  # 提高风险收益比
            print("   建议: 扩大盈利空间")
            print(f"   - ATR_TARGET_MULTIPLIER: 4.0")
            print(f"   - MIN_RISK_REWARD_RATIO: 2.5")
        
        # 规则3: 止损频繁 -> 放宽止损
        exit_reasons = analysis_result.get('exit_reasons', {})
        if exit_reasons.get('stop_loss', 0) > analysis_result.get('total_trades', 1) * 0.3:
            print("\n📊 止损过于频繁")
            suggestions['ATR_STOP_MULTIPLIER'] = 2.0  # 放宽止损
            print("   建议: 放宽止损距离")
            print(f"   - ATR_STOP_MULTIPLIER: 2.0")
        
        # 规则4: 持仓时间过长 -> 添加时间止损
        if analysis_result.get('avg_holding_days', 0) > 8:
            print("\n📊 平均持仓时间过长")
            print("   建议: 考虑添加时间止损规则（如7天未盈利则卖出）")
        
        print("\n" + "=" * 80)
        
        return suggestions
    
    def export_tuning_report(self, output_path: str = 'quantitative_retest/tuning_report.json'):
        """导出调优报告"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confidence_calibration': self.confidence_calibration,
            'parameter_impact': self.parameter_impact
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 调优报告已保存到: {output_path}")


# ========== 使用示例 ==========

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python smart_parameter_tuner.py <交易记录CSV路径>")
        print("示例: python smart_parameter_tuner.py backtest_trades_liquidity_grab_20260121_220321.csv")
        sys.exit(1)
    
    trades_csv = sys.argv[1]
    
    if not os.path.exists(trades_csv):
        print(f"错误: 文件不存在 - {trades_csv}")
        sys.exit(1)
    
    # 创建调优器
    tuner = SmartParameterTuner()
    
    # 分析历史交易
    analysis_result = tuner.analyze_historical_trades(trades_csv)
    
    # 尝试校准置信度（如果CSV中有置信度列）
    tuner.calibrate_confidence_from_trades(trades_csv)
    
    # 生成参数调整建议
    suggestions = tuner.suggest_parameter_adjustments(analysis_result)
    
    # 导出报告
    tuner.export_tuning_report()
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print("\n下一步:")
    print("1. 查看调优报告: quantitative_retest/tuning_report.json")
    print("2. 根据建议调整参数")
    print("3. 重新运行回测验证效果")
    print("=" * 80)
