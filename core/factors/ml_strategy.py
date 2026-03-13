"""
基于机器学习因子的量化策略

将机器学习模型集成到现有的策略框架中
可以作为独立策略使用，也可以与其他策略组合
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import Dict, Optional

from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
from core.factors.ml_factor_model import MLFactorModel
from core.data.data_fetcher import DataFetcher
from config.factor_config import FactorConfig, TrainingConfig


class MLFactorStrategy:
    """基于机器学习因子的量化策略"""
    
    def __init__(self, model_path: str = None):
        """
        初始化策略
        
        参数:
            model_path: 模型文件路径
        """
        self.factor_calculator = ComprehensiveFactorCalculator()
        self.data_fetcher = DataFetcher()
        self.model = None
        
        # 如果没有提供模型路径，使用默认路径
        if model_path is None:
            model_path = 'models/xgboost_factor_model.pkl'
        
        # 确保路径是绝对路径或相对于项目根目录
        if not os.path.isabs(model_path):
            # 获取项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, model_path)
        
        # 加载模型
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"警告: 模型文件不存在: {model_path}")
    
    def _get_stock_data_from_db(self, stock_code: str, days: int = 300) -> Optional[pd.DataFrame]:
        """
        从数据库获取股票数据
        
        参数:
            stock_code: 股票代码
            days: 获取天数
        
        返回:
            DataFrame: 股票数据
        """
        try:
            from datetime import datetime, timedelta
            from config import DATABASE_PATH
            import sqlite3
            
            # 计算日期范围
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # 从数据库查询
            conn = sqlite3.connect(DATABASE_PATH)
            query = '''
                SELECT date, open, high, low, close, volume, amount, turnover_rate
                FROM daily_data
                WHERE code = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(stock_code, start_date, end_date))
            conn.close()
            
            if df.empty:
                return None
            
            return df
            
        except Exception as e:
            print(f"从数据库获取{stock_code}数据失败: {e}")
            return None
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        try:
            self.model = MLFactorModel()
            self.model.load_model(model_path)
            print(f"已加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.model = None
    
    def screen_stock(self, stock_code: str, min_confidence: float = 60.0) -> Optional[Dict]:
        """
        筛选股票
        
        参数:
            stock_code: 股票代码
            min_confidence: 最小置信度阈值
        
        返回:
            信号字典或None
        """
        if self.model is None or not self.model.is_trained:
            # 模型未加载或未训练，返回None
            return None
        
        try:
            # 获取历史数据（使用数据库）
            data = self._get_stock_data_from_db(stock_code, days=300)
            
            if data is None or len(data) < 100:
                return None
            
            # 计算因子
            factors = self.factor_calculator.calculate_all_factors(
                stock_code, 
                data, 
                apply_feature_engineering=True,
                target_features=self.model.feature_names if self.model else None
            )
            
            if factors is None or len(factors) == 0 or factors.shape[1] == 0:
                # 因子计算失败或为空
                return None
            
            # 检查是否有NaN值
            if factors.isna().all().any():
                # 所有行都是NaN，跳过
                return None
            
            # 使用最新的因子数据进行预测
            latest_factors = factors.tail(1)
            
            # 检查最新因子是否有效
            if latest_factors.isna().all().any():
                return None
            
            # 生成信号
            signal_result = self.model.predict_signal(latest_factors)
            
            # 检查置信度
            if signal_result['confidence'] < min_confidence:
                return None
            
            # 计算止损和目标价
            current_price = data['close'].iloc[-1]
            atr = self._calculate_atr(data)
            
            stop_loss = current_price - 1.5 * atr
            target = current_price + 3 * atr
            
            # 获取因子重要性（用于分析）
            top_factors = self.model.get_top_factors(n=10)
            
            return {
                'stock_code': stock_code,
                'signal': signal_result['signal'],
                'confidence': signal_result['confidence'],
                'current_price': current_price,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'target': target,
                'prediction': signal_result['prediction'],
                'top_factors': top_factors,
                'analysis': {
                    'strategy_type': 'ml_factor',
                    'model_type': self.model.model_type,
                    'factor_count': len(self.model.feature_names)
                }
            }
        
        except Exception as e:
            # 静默处理错误，不打印
            return None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = None) -> float:
        """计算ATR"""
        if period is None:
            from config.strategy_config import ATR_PERIOD
            period = ATR_PERIOD
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if len(atr) > 0 else 0
    
    def batch_screen(self, stock_codes: list, min_confidence: float = 60.0) -> list:
        """
        批量筛选股票
        
        参数:
            stock_codes: 股票代码列表
            min_confidence: 最小置信度
        
        返回:
            符合条件的股票列表
        """
        results = []
        
        print(f"开始批量筛选 {len(stock_codes)} 只股票...")
        
        for i, code in enumerate(stock_codes, 1):
            if i % 50 == 0:
                print(f"  进度: {i}/{len(stock_codes)}")
            
            result = self.screen_stock(code, min_confidence)
            if result:
                results.append(result)
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\n筛选完成，找到 {len(results)} 只符合条件的股票")
        
        return results
    
    def analyze_factors(self, stock_code: str) -> Optional[Dict]:
        """
        分析股票的因子特征
        
        参数:
            stock_code: 股票代码
        
        返回:
            因子分析结果
        """
        try:
            # 获取数据（使用数据库）
            data = self._get_stock_data_from_db(stock_code, days=300)
            
            if data is None or len(data) < 100:
                return None
            
            # 计算因子
            factors = self.factor_calculator.calculate_all_factors(
                stock_code, 
                data, 
                apply_feature_engineering=True,
                target_features=self.model.feature_names if self.model else None
            )
            
            if factors is None or len(factors) == 0 or factors.shape[1] == 0:
                return None
            
            # 获取最新因子值
            latest_factors = factors.iloc[-1]
            
            # 获取因子重要性
            feature_importance = self.model.feature_importance if self.model else {}
            
            # 分析每个因子
            factor_analysis = []
            for factor_name, factor_value in latest_factors.items():
                importance = feature_importance.get(factor_name, 0)
                
                factor_analysis.append({
                    'name': factor_name,
                    'value': float(factor_value),
                    'importance': float(importance),
                    'contribution': float(factor_value * importance)
                })
            
            # 按重要性排序
            factor_analysis.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'stock_code': stock_code,
                'date': data['date'].iloc[-1],
                'factors': factor_analysis[:20],  # 返回前20个
                'total_factors': len(factor_analysis)
            }
        
        except Exception as e:
            # 静默处理错误
            return None


class HybridStrategy:
    """混合策略 - 结合机器学习和传统技术分析"""
    
    def __init__(self, ml_model_path: str, traditional_strategy):
        """
        初始化混合策略
        
        参数:
            ml_model_path: ML模型路径
            traditional_strategy: 传统策略对象（如SMCLiquidityStrategy）
        """
        self.ml_strategy = MLFactorStrategy(ml_model_path)
        self.traditional_strategy = traditional_strategy
    
    def screen_stock(self, stock_code: str, 
                    ml_weight: float = 0.5, 
                    traditional_weight: float = 0.5) -> Optional[Dict]:
        """
        混合筛选
        
        参数:
            stock_code: 股票代码
            ml_weight: ML策略权重
            traditional_weight: 传统策略权重
        
        返回:
            综合信号
        """
        # ML策略信号
        ml_result = self.ml_strategy.screen_stock(stock_code, min_confidence=50)
        
        # 传统策略信号
        traditional_result = self.traditional_strategy.screen_stock(stock_code)
        
        # 如果两个策略都没有信号，返回None
        if ml_result is None and traditional_result is None:
            return None
        
        # 如果只有一个策略有信号
        if ml_result is None:
            return traditional_result
        if traditional_result is None:
            return ml_result
        
        # 两个策略都有信号，进行加权融合
        ml_conf = ml_result['confidence']
        trad_conf = traditional_result['confidence']
        
        # 综合置信度
        combined_confidence = ml_conf * ml_weight + trad_conf * traditional_weight
        
        # 综合信号（两个都是buy才是buy）
        if ml_result['signal'] == 'buy' and traditional_result['signal'] in ['buy', 'strong_buy']:
            combined_signal = 'buy'
        else:
            combined_signal = 'hold'
        
        # 使用传统策略的价格信息（更保守）
        return {
            'stock_code': stock_code,
            'signal': combined_signal,
            'confidence': combined_confidence,
            'current_price': traditional_result['current_price'],
            'entry_price': traditional_result['entry_price'],
            'stop_loss': traditional_result['stop_loss'],
            'target': traditional_result['target'],
            'analysis': {
                'strategy_type': 'hybrid',
                'ml_confidence': ml_conf,
                'traditional_confidence': trad_conf,
                'ml_prediction': ml_result.get('prediction', 0),
                'top_factors': ml_result.get('top_factors', [])
            }
        }


def demo():
    """演示如何使用ML因子策略"""
    print("="*80)
    print("机器学习因子策略演示")
    print("="*80)
    
    # 1. 加载模型
    model_path = 'models/xgboost_factor_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行 train_ml_model.py 训练模型")
        return
    
    strategy = MLFactorStrategy(model_path)
    
    # 2. 测试单只股票
    test_code = '000001.SZ'
    print(f"\n测试股票: {test_code}")
    
    result = strategy.screen_stock(test_code, min_confidence=60)
    
    if result:
        print(f"\n信号: {result['signal']}")
        print(f"置信度: {result['confidence']:.2f}%")
        print(f"当前价格: {result['current_price']:.2f}")
        print(f"止损价: {result['stop_loss']:.2f}")
        print(f"目标价: {result['target']:.2f}")
        print(f"\n前10个重要因子:")
        for i, (name, importance) in enumerate(result['top_factors'][:10], 1):
            print(f"  {i}. {name}: {importance:.4f}")
    else:
        print("未找到买入信号")
    
    # 3. 因子分析
    print(f"\n{'='*80}")
    print("因子分析")
    print(f"{'='*80}")
    
    analysis = strategy.analyze_factors(test_code)
    
    if analysis:
        print(f"\n股票: {analysis['stock_code']}")
        print(f"日期: {analysis['date']}")
        print(f"\n前10个重要因子及其值:")
        for i, factor in enumerate(analysis['factors'][:10], 1):
            print(f"  {i}. {factor['name']}: {factor['value']:.4f} "
                  f"(重要性: {factor['importance']:.4f})")


if __name__ == '__main__':
    demo()
