"""
ML因子策略（回测版本）

将ML因子模型集成到新的回测框架
回测时完全依赖训练阶段生成的因子缓存，不再实时计算特征工程
"""

import os
import sys
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core.backtest.strategy import BaseStrategy, StrategySignal
from core.factors.ml_factor_model import MLFactorModel
from config.factor_config import TrainingConfig, FactorConfig
from config.strategy_config import ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER, MAX_POSITIONS


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
        生成交易信号 (批量预测优化版)
        """
        signals = []
        
        # 获取当前持仓和可用头寸
        existing_positions = portfolio_state.get('positions', {})
        current_count = len(existing_positions)
        available_slots = MAX_POSITIONS - current_count
        
        if available_slots <= 0:
            return signals
        
        # 1. 收集所有符合条件的股票及其最新因子
        valid_candidates = []
        
        # if current_date < "2025-02-20": # 仅在回测开始的前几天打印一次总况
        #      print(f"  [DEBUG] {current_date}: 正在检查 {len(market_data)} 只股票的市场数据...")

        for stock_code, stock_data in market_data.items():
            # 过滤已持有的股票，避免重复开仓
            if stock_code in existing_positions:
                continue
                
            if len(stock_data) < 35:
                continue
            
            try:
                # 获取因子（从缓存）
                factors = self._get_factors(stock_code, stock_data, current_date)
                
                if factors is None or len(factors) == 0:
                    continue
                
                # 检查最新因子日期是否与当前日期匹配
                latest_factor_date = factors['date'].iloc[-1] if 'date' in factors.columns else None
                if latest_factor_date:
                    latest_date_str = str(latest_factor_date)[:10]
                    if latest_date_str != current_date:
                        continue
                
                # 使用最新因子
                latest_factors = factors.tail(1).copy()
                if latest_factors.isna().all().any():
                    continue
                
                valid_candidates.append({
                    'code': stock_code,
                    'factors': latest_factors,
                    'data': stock_data
                })
            except Exception:
                continue
        
        if not valid_candidates:
            return signals
            
        # 2. 批量预测
        all_factors_list = [c['factors'] for c in valid_candidates]
        X_batch = pd.concat(all_factors_list, axis=0, ignore_index=True)
        
        # 移除日期列
        if 'date' in X_batch.columns:
            X_batch = X_batch.drop(columns=['date'])
            
        # 批量获取预测得分/置信度
        try:
            # --- 关键修复：每日横向 Z-Score 归一化 ---
            # 确保回测时的特征量纲与训练阶段（train_ml_model.py L549）完全一致
            if len(X_batch) > 1:
                # 减去均值，除以标准差。如果标准差为0则替换为1以避免除零，最后填充NaN为0
                X_batch_norm = (X_batch - X_batch.mean()) / X_batch.std().replace(0, 1.0)
                X_batch_norm = X_batch_norm.fillna(0)
            else:
                # 如果只有一只样本，无法计算标准差，统一设为 0（代表均值水平）
                X_batch_norm = X_batch * 0
                
            probs = self.model.predict(X_batch_norm)
        except Exception as e:
            print(f"  错误: 批量预测失败: {e}")
            return signals
            
        # 3. 筛选并构造候选列表
        candidates = []
        for i, candidate in enumerate(valid_candidates):
            prob = probs[i]
            confidence = float(prob * 100)
            
            # 基础过滤：置信度和信号阈值
            if confidence < self.min_confidence:
                continue
                
            # 改进：如果是排序任务，取消 0.5 (optimal_threshold) 的硬性过滤
            # 因为排序模型的输出是相对分数，整体位移可能导致所有分值略低于或高于 0.5
            if self.model.task != 'ranking':
                if prob < self.model.optimal_threshold:
                    continue
                
            stock_code = candidate['code']
            stock_data = candidate['data']
            current_price = stock_data['close'].iloc[-1]
            
            # 计算止损止盈
            atr = self._calculate_atr(stock_data, period=FactorConfig.ATR_PERIOD)
            stop_loss = current_price - ATR_STOP_MULTIPLIER * atr
            take_profit = current_price + ATR_TARGET_MULTIPLIER * atr
            
            candidates.append({
                'stock_code': stock_code,
                'confidence': confidence, # 强制限制在100以内
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'prediction': prob
            })
            
        # 4. 选择最优信号 (使用置信度排序)
        if candidates:
            # 引入确定性随机扰动作为平局决胜
            import hashlib
            def get_tie_breaker(code):
                return int(hashlib.md5(code.encode()).hexdigest(), 16) % 1000 / 100000.0

            # 按置信度由高到低排序，置信度相同时使用哈希值平局决胜
            candidates.sort(key=lambda x: (-(x['confidence'] + get_tie_breaker(x['stock_code']))))
            
            # 选择前 available_slots 个最优信号
            top_candidates = candidates[:available_slots]
            
            # if current_date < "2025-03-01": # 仅在开始阶段输出
                # print(f"  [DEBUG] {current_date}: 候选股票 {len(candidates)} 只, 买入头寸: {len(top_candidates)}")
            
            for cand in top_candidates:
                signal = StrategySignal(
                    stock_code=cand['stock_code'],
                    signal_type='buy',
                    timestamp=current_date,
                    price=cand['current_price'],
                    confidence=cand['confidence'],
                    stop_loss=cand['stop_loss'],
                    take_profit=cand['take_profit'],
                    metadata={
                        'strategy': 'ml_factor',
                        'model_type': self.model.model_type,
                        'prediction': cand['prediction'],
                        'confidence': cand['confidence'],
                        'candidates_count': len(candidates)
                    }
                )
                signals.append(signal)
            
        return signals
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        使用 talib 计算 ATR，确保与训练逻辑一致
        
        参数:
            data: 股票数据 DataFrame
            period: 周期
            
        返回:
            ATR 值
        """
        if len(data) < period + 1:
            return 0.0
            
        try:
            atr_series = talib.ATR(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                timeperiod=period
            )
            val = atr_series[-1]
            return float(val) if np.isfinite(val) else 0.0
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
            # 确保日期列为 datetime 类型
            if not pd.api.types.is_datetime64_any_dtype(cached_factors['date']):
                cached_factors['date'] = pd.to_datetime(cached_factors['date'])
            
            # 找到当前日期及之前的所有缓存
            target_dt = pd.Timestamp(current_date)
            factors = cached_factors[cached_factors['date'] <= target_dt].copy()
            if factors.empty:
                return None
        
        # 2. 如果缓存中没有日期列，则尝试使用行号对齐（不推荐，极易出错）
        else:
            raise ValueError("缓存中没有日期列，无法对齐")
        
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
