"""
K线形态检测模块 - 通用K线形态识别
供所有策略共享使用

包含常见的看涨和看空K线形态：
- 单根K线形态：锤子线、射击之星、十字星等
- 双根K线形态：吞没形态、乌云盖顶、刺透形态等
- 三根K线形态：晨星、暮星、三只乌鸦、三个白兵等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class CandlestickPatterns:
    """K线形态检测器"""
    
    def __init__(self):
        """初始化K线形态检测器"""
        pass
    
    # ==================== 单根K线形态 ====================
    
    def detect_hammer(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测锤子线形态（看涨反转）
        
        特征：
        - 下影线至少是实体的2倍
        - 上影线很小（<实体的10%）
        - 出现在下跌趋势底部
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 1:
            return None
        
        last = data.iloc[-1]
        body = abs(last['close'] - last['open'])
        lower_shadow = min(last['open'], last['close']) - last['low']
        upper_shadow = last['high'] - max(last['open'], last['close'])
        
        if body == 0:
            return None
        
        if lower_shadow >= 1.8 * body and upper_shadow <= 0.15 * body:
            return {
                'detected': True,
                'score': 20,
                'description': '锤子线',
                'type': 'bullish_reversal',
                'reliability': 75
            }
        return None
    
    def detect_shooting_star(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测射击之星形态（看跌反转）
        
        特征：
        - 上影线至少是实体的2倍
        - 出现在上涨趋势顶部
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 1:
            return None
        
        last = data.iloc[-1]
        body = abs(last['close'] - last['open'])
        lower_shadow = min(last['open'], last['close']) - last['low']
        upper_shadow = last['high'] - max(last['open'], last['close'])
        
        if body == 0:
            return None
        
        if upper_shadow > body * 2:
            return {
                'detected': True,
                'score': min(35, int(upper_shadow / body * 10)),
                'description': '射击之星',
                'type': 'bearish_reversal',
                'reliability': 70
            }
        return None
    
    def detect_doji(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测十字星形态（犹豫信号）
        
        特征：
        - 实体很小（<总区间的5%）
        - 开盘价和收盘价几乎相等
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 1:
            return None
        
        last = data.iloc[-1]
        body = abs(last['close'] - last['open'])
        total_range = last['high'] - last['low']
        
        if total_range == 0:
            return None
        
        if body <= 0.05 * total_range:
            return {
                'detected': True,
                'score': 10,
                'description': '十字星',
                'type': 'indecision',
                'reliability': 60
            }
        return None
    
    def detect_spinning_top(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测陀螺线形态（犹豫信号）
        
        特征：
        - 实体较小
        - 上下影线都较长
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 1:
            return None
        
        last = data.iloc[-1]
        body = abs(last['close'] - last['open'])
        lower_shadow = min(last['open'], last['close']) - last['low']
        upper_shadow = last['high'] - max(last['open'], last['close'])
        
        if body == 0:
            return None
        
        if (lower_shadow > body and upper_shadow > body and
            lower_shadow < body * 3 and upper_shadow < body * 3):
            return {
                'detected': True,
                'score': 8,
                'description': '陀螺线',
                'type': 'indecision',
                'reliability': 50
            }
        return None
    
    # ==================== 双根K线形态 ====================
    
    def detect_bullish_engulfing(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测看涨吞没形态
        
        特征：
        - 前一根是阴线
        - 当前是阳线
        - 当前K线完全吞没前一根
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 2:
            return None
        
        prev = data.iloc[-2]
        curr = data.iloc[-1]
        
        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])
        
        if (prev['close'] < prev['open'] and  # 前一根是阴线
            curr['close'] > curr['open'] and  # 当前是阳线
            curr['open'] < prev['close'] and  # 当前开盘低于前一收盘
            curr['close'] > prev['open'] and  # 当前收盘高于前一开盘
            curr_body > prev_body):           # 当前实体更大
            
            return {
                'detected': True,
                'score': 25,
                'description': '看涨吞没',
                'type': 'bullish_reversal',
                'reliability': 78
            }
        return None
    
    def detect_bearish_engulfing(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测看跌吞没形态
        
        特征：
        - 前一根是阳线
        - 当前是阴线
        - 当前K线完全吞没前一根
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 2:
            return None
        
        prev = data.iloc[-2]
        curr = data.iloc[-1]
        
        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])
        
        if (prev['close'] > prev['open'] and  # 前一根是阳线
            curr['close'] < curr['open'] and  # 当前是阴线
            curr['open'] > prev['close'] and  # 当前开盘高于前一收盘
            curr['close'] < prev['open'] and  # 当前收盘低于前一开盘
            curr_body > prev_body):           # 当前实体更大
            
            return {
                'detected': True,
                'score': 25,
                'description': '看跌吞没',
                'type': 'bearish_reversal',
                'reliability': 78
            }
        return None
    
    def detect_dark_cloud_cover(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测乌云盖顶形态（看跌反转）
        
        特征：
        - 前一根是大阳线
        - 当前是阴线
        - 跳空高开
        - 收盘在前一根中点下方
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 2:
            return None
        
        prev = data.iloc[-2]
        curr = data.iloc[-1]
        
        if (prev['close'] > prev['open'] and  # 前一根是阳线
            curr['close'] < curr['open'] and  # 当前是阴线
            curr['open'] > prev['high'] and  # 跳空高开
            curr['close'] < (prev['open'] + prev['close']) / 2):  # 收盘在前一根中点下方
            
            return {
                'detected': True,
                'score': 25,
                'description': '乌云盖顶',
                'type': 'bearish_reversal',
                'reliability': 75
            }
        return None
    
    def detect_piercing_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测刺透形态（看涨反转）
        
        特征：
        - 前一根是大阴线
        - 当前是阳线
        - 跳空低开
        - 收盘在前一根中点上方
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 2:
            return None
        
        prev = data.iloc[-2]
        curr = data.iloc[-1]
        
        if (prev['close'] < prev['open'] and  # 前一根是阴线
            curr['close'] > curr['open'] and  # 当前是阳线
            curr['open'] < prev['low'] and  # 跳空低开
            curr['close'] > (prev['open'] + prev['close']) / 2):  # 收盘在前一根中点上方
            
            return {
                'detected': True,
                'score': 25,
                'description': '刺透形态',
                'type': 'bullish_reversal',
                'reliability': 75
            }
        return None
    
    # ==================== 三根K线形态 ====================
    
    def detect_three_black_crows(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测三只乌鸦形态（强烈看跌）
        
        特征：
        - 三根连续阴线
        - 收盘价逐渐走低
        - 每根开盘在前一根实体内
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 3:
            return None
        
        last_three = data.tail(3)
        
        # 检查是否都是阴线且逐渐走低
        all_bearish = all(row['close'] < row['open'] for _, row in last_three.iterrows())
        descending = (last_three.iloc[0]['close'] > last_three.iloc[1]['close'] > 
                     last_three.iloc[2]['close'])
        
        if all_bearish and descending:
            return {
                'detected': True,
                'score': 30,
                'description': '三只乌鸦',
                'type': 'bearish_reversal',
                'reliability': 78
            }
        return None
    
    def detect_three_white_soldiers(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测三个白兵形态（强烈看涨）
        
        特征：
        - 三根连续阳线
        - 收盘价逐渐走高
        - 每根开盘在前一根实体内
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 3:
            return None
        
        last_three = data.tail(3)
        
        # 检查是否都是阳线且逐渐走高
        all_bullish = all(row['close'] > row['open'] for _, row in last_three.iterrows())
        ascending = (last_three.iloc[0]['close'] < last_three.iloc[1]['close'] < 
                    last_three.iloc[2]['close'])
        
        if all_bullish and ascending:
            return {
                'detected': True,
                'score': 30,
                'description': '三个白兵',
                'type': 'bullish_reversal',
                'reliability': 78
            }
        return None
    
    def detect_morning_star(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测晨星形态（看涨反转）
        
        特征：
        - 第一根：大阴线
        - 第二根：小实体（十字星或小K线）
        - 第三根：大阳线，深入第一根实体
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 3:
            return None
        
        bar1 = data.iloc[-3]
        bar2 = data.iloc[-2]
        bar3 = data.iloc[-1]
        
        # 第一根：大阴线
        is_bearish_1 = bar1['close'] < bar1['open'] and (bar1['open'] - bar1['close']) / bar1['open'] > 0.02
        # 第二根：小实体
        is_small_2 = abs(bar2['close'] - bar2['open']) / bar2['open'] < 0.01
        # 第三根：大阳线
        is_bullish_3 = bar3['close'] > bar3['open'] and (bar3['close'] - bar3['open']) / bar3['open'] > 0.02
        # 第三根深入第一根
        penetration = bar3['close'] > (bar1['open'] + bar1['close']) / 2
        
        if is_bearish_1 and is_small_2 and is_bullish_3 and penetration:
            return {
                'detected': True,
                'score': 25,
                'description': '晨星',
                'type': 'bullish_reversal',
                'reliability': 68
            }
        return None
    
    def detect_evening_star(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        检测黄昏之星/暮星形态（看跌反转）
        
        特征：
        - 第一根：大阳线
        - 第二根：小实体（十字星或小K线）
        - 第三根：大阴线，深入第一根实体
        
        返回: {'detected': bool, 'score': int, 'description': str}
        """
        if len(data) < 3:
            return None
        
        bar1 = data.iloc[-3]
        bar2 = data.iloc[-2]
        bar3 = data.iloc[-1]
        
        # 第一根：大阳线
        is_bullish_1 = bar1['close'] > bar1['open'] and (bar1['close'] - bar1['open']) / bar1['open'] > 0.02
        # 第二根：小实体
        is_small_2 = abs(bar2['close'] - bar2['open']) / bar2['open'] < 0.01
        # 第三根：大阴线
        is_bearish_3 = bar3['close'] < bar3['open'] and (bar3['open'] - bar3['close']) / bar3['open'] > 0.02
        # 第三根深入第一根
        penetration = bar3['close'] < (bar1['open'] + bar1['close']) / 2
        
        if is_bullish_1 and is_small_2 and is_bearish_3 and penetration:
            return {
                'detected': True,
                'score': 25,
                'description': '黄昏之星',
                'type': 'bearish_reversal',
                'reliability': 68
            }
        return None
    
    # ==================== 综合检测方法 ====================
    
    def detect_all_bullish_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        检测所有看涨形态
        
        返回: 检测到的形态列表
        """
        patterns = []
        
        # 单根K线形态
        hammer = self.detect_hammer(data)
        if hammer:
            patterns.append(hammer)
        
        # 双根K线形态
        bullish_engulfing = self.detect_bullish_engulfing(data)
        if bullish_engulfing:
            patterns.append(bullish_engulfing)
        
        piercing = self.detect_piercing_pattern(data)
        if piercing:
            patterns.append(piercing)
        
        # 三根K线形态
        three_soldiers = self.detect_three_white_soldiers(data)
        if three_soldiers:
            patterns.append(three_soldiers)
        
        morning_star = self.detect_morning_star(data)
        if morning_star:
            patterns.append(morning_star)
        
        return patterns
    
    def detect_all_bearish_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        检测所有看跌形态
        
        返回: 检测到的形态列表
        """
        patterns = []
        
        # 单根K线形态
        shooting_star = self.detect_shooting_star(data)
        if shooting_star:
            patterns.append(shooting_star)
        
        # 双根K线形态
        bearish_engulfing = self.detect_bearish_engulfing(data)
        if bearish_engulfing:
            patterns.append(bearish_engulfing)
        
        dark_cloud = self.detect_dark_cloud_cover(data)
        if dark_cloud:
            patterns.append(dark_cloud)
        
        # 三根K线形态
        three_crows = self.detect_three_black_crows(data)
        if three_crows:
            patterns.append(three_crows)
        
        evening_star = self.detect_evening_star(data)
        if evening_star:
            patterns.append(evening_star)
        
        return patterns
    
    def detect_all_patterns(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        检测所有K线形态
        
        返回: {'bullish': [...], 'bearish': [...], 'neutral': [...]}
        """
        bullish = self.detect_all_bullish_patterns(data)
        bearish = self.detect_all_bearish_patterns(data)
        
        # 检测中性形态
        neutral = []
        doji = self.detect_doji(data)
        if doji:
            neutral.append(doji)
        
        spinning_top = self.detect_spinning_top(data)
        if spinning_top:
            neutral.append(spinning_top)
        
        return {
            'bullish': bullish,
            'bearish': bearish,
            'neutral': neutral
        }
    
    def get_total_bearish_score(self, data: pd.DataFrame) -> int:
        """
        获取看跌形态的总分
        
        用于策略中的看空信号评分
        """
        bearish_patterns = self.detect_all_bearish_patterns(data)
        return sum(pattern['score'] for pattern in bearish_patterns)
    
    def get_total_bullish_score(self, data: pd.DataFrame) -> int:
        """
        获取看涨形态的总分
        
        用于策略中的看多信号评分
        """
        bullish_patterns = self.detect_all_bullish_patterns(data)
        return sum(pattern['score'] for pattern in bullish_patterns)
