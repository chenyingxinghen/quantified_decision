"""
因子模型配置文件

包含：
1. 模型超参数配置
2. 训练参数配置
3. 因子计算参数配置
4. 优化参数配置
"""

from typing import Dict, Any
from config import baostock_config


# ============================================================================
# 1. 模型超参数配置
# ============================================================================

class ModelConfig:
    """模型超参数配置"""
    
    # XGBoost配置
    XGBOOST_PARAMS = {
        'n_estimators': 3000,
        'max_depth': 6,              # 增加深度以改善预测区分度
        'min_child_weight': 200,       # 增加权重要求，防止过拟合
        'learning_rate': 0.05,       
        'subsample': 1,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'gamma': 0.17,    
        
        'reg_alpha': 11,            
        'reg_lambda': 23,             
        'objective': 'reg:logistic', 
        'eval_metric': 'auc',       
        'n_jobs': 15,
        'early_stopping_rounds': 20,
        'verbosity': 0,              # 打印训练过程
    }
    
    # LightGBM配置
    LIGHTGBM_PARAMS = {
        'n_estimators': 3000,
        'max_depth': 6,
        'min_child_weight': 200,
        'num_leaves': 63,
        'learning_rate': 0.05,
        'min_gain_to_split': 0.1,

        'reg_alpha': 3,
        'reg_lambda': 7,
        'subsample': 1,
        'colsample_bytree': 1,

        'label_gain': [float(i**2 - 1) for i in range(100)], 
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 100,
        'n_jobs': 15,
        'verbosity': -1,
        'early_stopping_rounds': 50,
    }
    

    # GPU 专用配置增量 (如果 USE_GPU = True)
    GPU_PARAMS_XGB = {
        'tree_method': 'hist',      # XGBoost 2.0+ 推荐使用 hist + device=cuda
        'device': 'cuda',           # 显式指定使用 CUDA
    }
    
    
    
    @classmethod
    def get_model_params(cls, model_type: str) -> Dict[str, Any]:
        """获取指定模型的参数"""
        params_map = {
            'xgboost': cls.XGBOOST_PARAMS,
            'lightgbm': cls.LIGHTGBM_PARAMS,
        }
        params = params_map.get(model_type, {}).copy()
        
        # 如果启用了 GPU 加速
        if model_type == 'xgboost':
            params.update(cls.GPU_PARAMS_XGB)
                
        return params


# ============================================================================
# 2. 训练参数配置
# ============================================================================

class TrainingConfig:
    """训练参数配置"""
    # 模型训练任务类型 (LGBM固定为ranking, XGB固定为regression拟合软化标签)
    TASK_TYPE = 'hybrid' 
    MODEL_TYPES = ['lightgbm']


    INCLUDE_FUNDAMENTALS = True  # 是否包含基本面因子
    PUNISH_UNBUYABLE = True      # 涨停板、停牌样本惩罚
    USE_GPU = True               # 是否启用 GPU 加速

    YEARS=baostock_config.HISTORY_YEARS

    YEARS_FOR_BACKTEST=1         # 回测年数
    YEARS_FOR_TRAINING=15         # 训练年数
    STOCK_NUM = 6000             # 股票数量
    # 数据集划分
    TRAIN_TEST_SPLIT = 0.7
    

    # 预测天数 (用于分类、回归和排序任务)
    FUTURE_DAYS = 5
    

    # 缓存目录
    CACHE_DIR = 'database/system_data/factors_cache'
    # 模型保存目录
    SAVE_DIR = 'models'
    


# ============================================================================
# 3. 因子计算参数配置
# ============================================================================

class FactorConfig:
    """因子计算参数配置"""
    
    # 动量因子参数
    RSI_PERIOD = 42
    ROC_PERIOD = 60
    MTM_PERIOD = 40
    CMO_PERIOD = 42
    STOCHRSI_PERIOD = 28
    RVI_PERIOD = 14
    
    # 趋势因子参数
    MACD_FAST = 25
    MACD_SLOW = 90
    MACD_SIGNAL = 20
    ADX_PERIOD = 35
    DMI_PERIOD = 35
    AROON_PERIOD = 50
    TRIX_PERIOD = 30
    
    # 均线参数
    MA_RATIO_PERIOD = 120
    MA_SLOPE_PERIOD = 30
    
    # 波动率因子参数
    ATR_PERIOD = 30
    NATR_PERIOD = 28
    BB_PERIOD = 100
    BB_STD = 1.5
    CCI_PERIOD = 35
    ULCER_PERIOD = 35
    PRICE_VAR_PERIOD = 30
    
    # 成交量因子参数
    VOLUME_MA_PERIOD = 5
    VOLUME_STD_PERIOD = 10
    VOLUME_MA_SHORT = 5
    VOLUME_MA_LONG = 10
    AMOUNT_MA_PERIOD = 5
    AMOUNT_STD_PERIOD = 10
    MFI_PERIOD = 35
    VR_PERIOD = 52
    VROC_PERIOD = 36
    VRSI_PERIOD = 21
    VMACD_FAST = 25
    VMACD_SLOW = 60
    VMACD_SIGNAL = 20
    ADOSC_FAST = 3
    ADOSC_SLOW = 7
    
    # 摆动指标参数
    KDJ_N = 28
    WILLR_PERIOD = 35
    BIAS_PERIOD = 36
    PSY_PERIOD = 30
    AR_BR_PERIOD = 52
    CR_PERIOD = 52
    
    # K线形态参数
    BODY_SIZE_THRESHOLD_LARGE = 0.015  # 大实体阈值（2%）
    BODY_SIZE_THRESHOLD_SMALL = 0.003  # 小实体阈值（1%）
    HAMMER_LOWER_SHADOW_RATIO = 1.5   # 锤子线下影线/实体比率
    HAMMER_UPPER_SHADOW_RATIO = 0.5   # 锤子线上影线/实体比率


# ============================================================================
# 4. 优化参数配置
# ============================================================================

class OptimizationConfig:
    """优化参数配置"""
    
    # 特征选择方法
    FEATURE_SELECTION_METHOD = 'hybrid'  # 'importance', 'correlation', 'mutual_info', 'rfe', 'hybrid'
    N_FEATURES_TO_SELECT = 60  # 增加特征数，由 40 提高到 60，保留更多有潜在价值的因子
    
    # 特征选择阈值
    FEATURE_IMPORTANCE_THRESHOLD = 0.001
    CORRELATION_THRESHOLD = 0.95
    CORRELATION_THRESHOLD_LOW = 0.05
    
    # 因子参数优化设置
    FACTOR_TUNING_METRIC = 'ic'  # 'ic', 'rank_ic', 'auc'
    FACTOR_TUNING_METHOD = 'coordinate_descent'
    N_ITER = 50
    CV_FOLDS = 3
    
    # 集成学习优化
    ENSEMBLE_OPTIMIZATION_METHOD = 'grid'
    ENSEMBLE_GRID_RESOLUTION = 21
    USE_STACKING = False
    
    # 因子工程优化
    OPTIMIZE_FACTOR_PERIODS = False


# ============================================================================
# 5. 配置管理
# ============================================================================

class FactorModelConfig:
    """因子模型统一配置管理"""
    
    model = ModelConfig
    training = TrainingConfig
    factor = FactorConfig
    optimization = OptimizationConfig
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 80)
        print("因子模型配置信息")
        print("=" * 80)
        
        print("\n[模型配置]")
        print(f"XGBoost树数量: {cls.model.XGBOOST_PARAMS['n_estimators']}")
        print(f"学习率: {cls.model.XGBOOST_PARAMS['learning_rate']}")
        
        print("\n[训练配置]")
        print(f"训练集比例: {cls.training.TRAIN_TEST_SPLIT}")
        print(f"预测天数: {cls.training.FUTURE_DAYS}")
        
        print("\n[因子配置]")
        print(f"RSI周期: {cls.factor.RSI_PERIOD}")
        print(f"MACD参数: ({cls.factor.MACD_FAST}, {cls.factor.MACD_SLOW}, {cls.factor.MACD_SIGNAL})")
        print(f"ATR周期: {cls.factor.ATR_PERIOD}")
        
        print("\n[优化配置]")
        print(f"特征选择方法: {cls.optimization.FEATURE_SELECTION_METHOD}")
        print(f"选择特征数: {cls.optimization.N_FEATURES_TO_SELECT}")
        
        print("=" * 80)


if __name__ == '__main__':
    FactorModelConfig.print_config()
