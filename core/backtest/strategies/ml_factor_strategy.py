"""
ML因子策略（回测版本）

将ML因子模型集成到新的回测框架
回测时完全依赖训练阶段生成的因子缓存，不再实时计算特征工程
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core.backtest.strategy import BaseStrategy, StrategySignal
from core.factors.ml_factor_model import MLFactorModel
from config.factor_config import TrainingConfig, FactorConfig
from config.strategy_config import ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER


class MLFactorBacktestStrategy(BaseStrategy):
    """ML因子回测策略
    
    回测时完全依赖训练阶段生成的因子缓存。
    缓存中应包含模型需要的所有特征（含特征工程生成的特征）。
    如果缓存缺失某只股票，该股票将被跳过。
    如果缓存中缺少部分特征列，缺失列将被填充为0。
    """
    
    def __init__(self,
                 model_path: str,
                 min_confidence: float = 60.0,
                 use_cache: bool = True,
                 cache_dir: str = None,
                 name: str = "ML因子策略"):
        """
        初始化策略
        
        参数:
            model_path: 模型文件路径
            min_confidence: 最小置信度阈值
            use_cache: 是否使用因子缓存
            cache_dir: 缓存目录路径
            name: 策略名称
        """
        super().__init__(name)
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.use_cache = use_cache
        
        # 设置缓存目录
        if cache_dir is None:
            cache_dir = TrainingConfig.CACHE_DIR
        self.cache_dir = cache_dir
        
        self.model = None
        self._factors_cache = {}  # 内存缓存
        self._warned_stocks = set()  # 已警告过的股票，避免重复日志
    
    def initialize(self, **kwargs):
        """初始化策略"""
        super().initialize(**kwargs)
        
        # 加载模型
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        self.model = MLFactorModel()
        self.model.load_model(self.model_path)
        
        if not self.model.is_trained:
            raise ValueError("模型未训练")
        
        # 检测缓存状态
        cache_status = "未找到"
        if self.use_cache and os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.parquet')]
            cache_status = f"{len(cache_files)} 只股票"
            
            # 检查第一个缓存文件的特征是否与模型匹配
            if cache_files:
                try:
                    sample = pd.read_parquet(os.path.join(self.cache_dir, cache_files[0]))
                    missing = [f for f in self.model.feature_names if f not in sample.columns]
                    if missing:
                        print(f"  警告: 缓存缺少 {len(missing)} 个模型特征，将用0填充")
                    else:
                        print(f"  缓存特征与模型完全匹配 ✓")
                except Exception:
                    pass
        
        print(f"策略初始化完成: {self.name}")
        print(f"  模型: {self.model_path}")
        print(f"  模型特征数: {len(self.model.feature_names)}")
        print(f"  最小置信度: {self.min_confidence}%")
        print(f"  使用缓存: {self.use_cache}")
        if self.use_cache:
            print(f"  缓存目录: {self.cache_dir}")
            print(f"  缓存状态: {cache_status}")
        print(f"  实时特征工程: 禁用（完全依赖缓存）")
    
    def generate_signals(self,
                        current_date: str,
                        market_data: Dict[str, pd.DataFrame],
                        portfolio_state: Dict[str, Any]) -> List[StrategySignal]:
        """
        生成交易信号
        
        参数:
            current_date: 当前日期
            market_data: 市场数据
            portfolio_state: 投资组合状态
        
        返回:
            信号列表
        """
        signals = []
        
        # 如果已有持仓，不生成新信号
        if portfolio_state.get('position_count', 0) > 0:
            return signals
        
        # 筛选股票
        candidates = []
        
        for stock_code, stock_data in market_data.items():
            # 数据量检查
            if len(stock_data) < 100:
                continue
            
            try:
                # 获取因子（从缓存）
                factors = self._get_factors(stock_code, stock_data, current_date)
                
                if factors is None or len(factors) == 0:
                    continue
                
                # 检查最新因子日期是否与当前日期匹配（允许10天以内的差距以兼容非交易日，但最好是严格匹配）
                latest_factor_date = factors['date'].iloc[-1] if 'date' in factors.columns else None
                if latest_factor_date and latest_factor_date != current_date:
                    # 如果日期不匹配，说明缓存不包含今日因子，跳过
                    continue
                
                # 检查NaN
                if factors.isna().all().any():
                    continue
                
                # 使用最新因子预测
                latest_factors = factors.tail(1)
                if latest_factors.isna().all().any():
                    continue
                
                # 预测
                prediction = self.model.predict_signal(latest_factors)
                
                # 检查置信度
                if prediction['confidence'] < self.min_confidence:
                    continue
                
                # 检查信号类型
                if prediction['signal'] not in ['buy', 'strong_buy']:
                    continue
                
                # 获取当前价格
                current_price = stock_data['close'].iloc[-1]
                
                # 计算止损止盈
                atr = self._calculate_atr(stock_data, period=FactorConfig.ATR_PERIOD)
                stop_loss = current_price - ATR_STOP_MULTIPLIER * atr
                take_profit = current_price + ATR_TARGET_MULTIPLIER * atr
                
                candidates.append({
                    'stock_code': stock_code,
                    'confidence': prediction['confidence'],
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'prediction': prediction['prediction']
                })
            
            except Exception as e:
                # 只对前几只出错的股票打印错误详情，避免日志淹没
                if len(self._warned_stocks) < 10 and stock_code not in self._warned_stocks:
                    print(f"  警告: 生成 {stock_code} 信号时出错: {e}")
                    self._warned_stocks.add(stock_code)
                continue
        
        # 选择置信度最高的
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            
            signal = StrategySignal(
                stock_code=best['stock_code'],
                signal_type='buy',
                timestamp=current_date,
                price=best['current_price'],
                confidence=best['confidence'],
                stop_loss=best['stop_loss'],
                take_profit=best['take_profit'],
                metadata={
                    'strategy': 'ml_factor',
                    'model_type': self.model.model_type,
                    'prediction': best['prediction'],
                    'confidence': best['confidence']
                }
            )
            
            signals.append(signal)
        
        return signals
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        计算简单的 ATR（平均真实波幅）
        
        参数:
            data: 股票数据 DataFrame
            period: 周期
            
        返回:
            ATR 值
        """
        if len(data) < period + 1:
            # 数据不足，返回最近的波动估算
            if len(data) >= 2:
                return float((data['high'] - data['low']).mean())
            return 0.0
            
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # 计算 TR (True Range)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # 计算 ATR (简单移动平均)
            atr = np.mean(tr[-period:])
            return float(atr)
        except Exception:
            return 0.0
    
    def _get_factors(self, stock_code: str, stock_data: pd.DataFrame, current_date: str) -> Optional[pd.DataFrame]:
        """
        从缓存获取因子，不做任何实时计算
        
        参数:
            stock_code: 股票代码
            stock_data: 股票数据（仅用于确定行数以截取缓存）
            current_date: 当前日期
        
        返回:
            因子DataFrame，或None（如果缓存不存在）
        """
        # 从缓存加载
        if not self.use_cache:
            # 缓存被禁用，无法获取因子
            return None
        
        cached_factors = self._load_factors_from_cache(stock_code)
        if cached_factors is None:
            # 没有缓存，跳过该股票
            return None
        
        # 1. 尝试使用日期对齐（最准确，推荐）
        if 'date' in cached_factors.columns:
            # 找到当前日期及之前的所有缓存
            factors = cached_factors[cached_factors['date'] <= current_date].copy()
            if factors.empty:
                # 可能是回测起始日早于缓存起始日
                return None
        
        # 2. 如果缓存中没有日期列，则尝试使用行号对齐（不推荐，极易出错）
        else:
            data_len = len(stock_data)
            if len(cached_factors) >= data_len:
                factors = cached_factors.iloc[:data_len].copy()
            else:
                # 缓存行数不够，使用全部缓存（这种情况下大概率会出现日期错位）
                factors = cached_factors.copy()
        
        # 确保所有模型需要的特征列都存在
        if self.model and self.model.feature_names:
            missing_features = [f for f in self.model.feature_names if f not in factors.columns]
            if missing_features:
                # 用0填充缺失特征（与训练时的填充策略一致）
                if stock_code not in self._warned_stocks:
                    self._warned_stocks.add(stock_code)
                    if len(self._warned_stocks) <= 3:  # 只对前3只股票打印警告
                        print(f"  提示: {stock_code} 缓存缺少 {len(missing_features)} 个特征，用0填充")
                
                missing_df = pd.DataFrame(0.0, index=factors.index, columns=missing_features)
                factors = pd.concat([factors, missing_df], axis=1)
            
            # 只保留模型需要的特征，并包含日期列（用于验证对齐）
            feature_cols = self.model.feature_names.copy()
            if 'date' in factors.columns and 'date' not in feature_cols:
                feature_cols.append('date')
            
            available = [f for f in feature_cols if f in factors.columns]
            if len(available) == 0:
                return None
            factors = factors[available]
        
        return factors
    
    def _load_factors_from_cache(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        从缓存加载因子
        
        参数:
            stock_code: 股票代码
        
        返回:
            因子DataFrame或None
        """
        # 检查内存缓存
        if stock_code in self._factors_cache:
            return self._factors_cache[stock_code]
        
        # 从文件加载
        cache_file = os.path.join(self.cache_dir, f'{stock_code}_factors.parquet')
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            factors = pd.read_parquet(cache_file)
            
            # 缓存到内存
            self._factors_cache[stock_code] = factors
            
            return factors
        
        except Exception:
            return None
    
    def cleanup(self):
        """清理资源"""
        self.model = None
        self._factors_cache.clear()
        self._warned_stocks.clear()
        print(f"策略清理完成: {self.name}")
