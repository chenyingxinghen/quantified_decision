"""
core.neural — 神经网络因子模型独立模块

设计目标：
- 完全复用既有数据基础设施（core.factors.train_ml_model.MLModelTrainer 的
  因子缓存、多目标标签、横截面归一化、特征选择），只替换"模型"本身。
- 对外暴露与 core.factors.ml_factor_model 完全一致的接口
  （predict / feature_names / predict_components / models / save_model / load_model），
  因此可直接被现有回测（MLFactorBacktestStrategy）与实盘选股（select_for_live）消费。
"""
from core.neural.nn_models import (
    NeuralNet,
    NeuralNetFactorModel,
    MultiObjectiveNeuralModel,
)
from core.neural.trainer import NeuralTrainer
from core.neural.portfolio import PortfolioOptimizer, select_portfolio

__all__ = [
    "NeuralNet",
    "NeuralNetFactorModel",
    "MultiObjectiveNeuralModel",
    "NeuralTrainer",
    "PortfolioOptimizer",
    "select_portfolio",
]
