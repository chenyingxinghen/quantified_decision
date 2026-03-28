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
    
    def _dynamic_adjust(self, df: pd.DataFrame) -> pd.DataFrame:
        """动态复权修正价格序列跳变"""
        if df is None or df.empty or 'fore_adjust_factor' not in df.columns:
            return df
        
        valid_adj = df['fore_adjust_factor'].dropna()
        if not valid_adj.empty:
            base_val = float(valid_adj.iloc[-1])
            if base_val != 0:
                ratio = df['fore_adjust_factor'].ffill().fillna(1.0) / base_val
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = df[col] * ratio
        return df

    def _get_stock_data_from_db(self, stock_code: str, days: int = 300) -> Optional[pd.DataFrame]:
        """
        从数据库获取股票数据 (修正：集成动态复权)
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
                SELECT k.date, k.open, k.high, k.low, k.close, k.volume, k.amount, k.turnover_rate, a.fore_adjust_factor
                FROM daily_data k
                LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
                WHERE k.code = ? AND k.date >= ? AND k.date <= ?
                ORDER BY k.date ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(stock_code, start_date, end_date))
            conn.close()
            
            if df.empty:
                return None
            
            return self._dynamic_adjust(df)
            
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
            import traceback
            print(f"  [ERROR] 筛选股票 {stock_code} 失败: {e}")
            traceback.print_exc()
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
        批量筛选股票 (修正：集成真正的横截面排名逻辑)
        
        流程:
        1. 获取所有股票的最新因子数据
        2. 将所有因子合并为大矩阵
        3. 执行每日横截面分位数标准化
        4. 模型批量预测 (保证 Confidence 的准确分布)
        """
        if self.model is None or not self.model.is_trained:
            return []
            
        print(f"\n开始批量筛选 {len(stock_codes)} 只股票 [集成横截面归一化]...")
        
        all_latest_factors = []
        valid_codes = []
        stock_prices = {}
        stock_atrs = {}

        # 1. 批量收集全量因子
        for i, code in enumerate(stock_codes):
            try:
                data = self._get_stock_data_from_db(code, days=300)
                if data is None or len(data) < 100: continue
                
                factors = self.factor_calculator.calculate_all_factors(
                    code, data, apply_feature_engineering=True,
                    target_features=self.model.feature_names
                )
                
                if factors is not None and not factors.empty:
                    # 取最新一行
                    latest = factors.iloc[[-1]].copy()
                    if not latest.isna().all().any():
                        all_latest_factors.append(latest)
                        valid_codes.append(code)
                        stock_prices[code] = data['close'].iloc[-1]
                        stock_atrs[code] = self._calculate_atr(data)
                
                if (i+1) % 50 == 0:
                    print(f"  收集因子进度: {i+1}/{len(stock_codes)}")
            except:
                continue

        if not all_latest_factors:
            return []

        # 2. 合并并进行横截面归一化
        all_X = pd.concat(all_latest_factors, axis=0, ignore_index=True)
        all_X = all_X.astype(np.float64)
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # 同步 train_ml_model 豁免逻辑
            sentiment_keys = ['up_ratio', 'down_ratio', 'mean_return', 'adv_vol', 'breadth_', 'sentiment_', 'mkt_', 'market_type']
            rank_cols = [col for col in all_X.columns if not any(k in col.lower() for k in sentiment_keys)]
            
            if rank_cols and len(all_X) > 1:
                all_X[rank_cols] = all_X[rank_cols].rank(pct=True).fillna(0.5)
        
        all_X = all_X.fillna(0.5)

        # 3. 批量预测
        print(f"  正在执行批量预测 [数量: {len(valid_codes)}]...")
        probs = self.model.predict(all_X)
        
        # 4. 汇总结果
        results = []
        top_factor_names = self.model.get_top_factors(n=10)

        for i, code in enumerate(valid_codes):
            prob = float(probs[i])
            confidence = prob * 100
            
            if confidence >= min_confidence:
                current_price = stock_prices[code]
                atr = stock_atrs[code]
                
                results.append({
                    'stock_code': code,
                    'signal': 'buy' if prob > 0.5 else 'hold',
                    'confidence': confidence,
                    'current_price': current_price,
                    'entry_price': current_price,
                    'stop_loss': current_price - 1.5 * atr,
                    'target': current_price + 3.0 * atr,
                    'prediction': prob,
                    'top_factors': top_factor_names,
                    'analysis': {
                        'strategy_type': 'ml_factor_integrated',
                        'model_type': self.model.model_type
                    }
                })
        
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
