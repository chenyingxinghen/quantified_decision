"""
因子模型配置文件

包含：
1. 模型超参数配置
2. 训练参数配置
3. 因子计算参数配置
4. 优化参数配置
"""

from typing import Dict, Any


# ============================================================================
# 1. 模型超参数配置
# ============================================================================

class ModelConfig:
    """模型超参数配置"""
    
    # XGBoost配置
    XGBOOST_PARAMS = {
        'n_estimators': 3000,
        'max_depth': 5,              # 增加深度以改善预测区分度
        'min_child_weight': 1,       # 增加权重要求，防止过拟合
        'learning_rate': 0.02,       # 降低学习率
        'subsample': 0.8,            
        'colsample_bytree': 0.5,     
        'reg_alpha': 3,            
        'reg_lambda': 8,             
        'objective': 'reg:logistic', 
        'eval_metric': 'auc',       
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 100,
    }
    
    # LightGBM配置
    LIGHTGBM_PARAMS = {
        'n_estimators': 3000,
        'max_depth': 4,
        'num_leaves': 31,
        'learning_rate': 0.02,
        'min_data_in_leaf': 150,
        'min_gain_to_split': 0.01,
        'reg_alpha': 3,
        'reg_lambda': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,

        'label_gain': [float(i**2 - 1) for i in range(100)], 
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 100,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'early_stopping_rounds': 100,
        'force_row_wise': True, 
    }
    
    
    # 默认使用的模型类型
    DEFAULT_MODELS = ['xgboost', 'lightgbm']
    
    @classmethod
    def get_model_params(cls, model_type: str) -> Dict[str, Any]:
        """获取指定模型的参数"""
        params_map = {
            'xgboost': cls.XGBOOST_PARAMS,
            'lightgbm': cls.LIGHTGBM_PARAMS,
        }
        return params_map.get(model_type, {}).copy()


# ============================================================================
# 2. 训练参数配置
# ============================================================================

class TrainingConfig:
    """训练参数配置"""
    # 模型训练任务类型 (LGBM固定为ranking, XGB固定为regression拟合软化标签)
    TASK_TYPE = 'hybrid' 

    INCLUDE_FUNDAMENTALS = False  # 是否包含基本面因子
    PUNISH_UNBUYABLE = True      # 涨停板、停牌样本惩罚

    YEARS_FOR_BACKTEST=1         # 回测年数
    YEARS_FOR_TRAINING=8         # 训练年数
    STOCK_NUM = 5000             # 股票数量
    # 数据集划分
    TRAIN_TEST_SPLIT = 0.8

    # 预测天数 (用于分类、回归和排序任务)
    FUTURE_DAYS = 10
    

    # 缓存目录
    CACHE_DIR = 'data/factors_cache'
    # 模型保存目录
    SAVE_DIR = 'models'
    


# ============================================================================
# 3. 因子计算参数配置
# ============================================================================

class FactorConfig:
    """因子计算参数配置"""
    
    # 动量因子参数
    RSI_PERIOD = 35
    ROC_PERIOD = 20
    MTM_PERIOD = 20
    CMO_PERIOD = 35
    STOCHRSI_PERIOD = 21
    RVI_PERIOD = 14
    
    # 趋势因子参数
    MACD_FAST = 20
    MACD_SLOW = 60
    MACD_SIGNAL = 15
    ADX_PERIOD = 7
    DMI_PERIOD = 14
    AROON_PERIOD = 30
    TRIX_PERIOD = 30
    
    # 均线参数
    MA_RATIO_PERIOD = 30
    MA_SLOPE_PERIOD = 10
    
    # 波动率因子参数
    ATR_PERIOD = 10
    NATR_PERIOD = 5
    BB_PERIOD = 50
    BB_STD = 1.5
    CCI_PERIOD = 28
    ULCER_PERIOD = 21
    PRICE_VAR_PERIOD = 10
    
    # 成交量因子参数
    VOLUME_MA_PERIOD = 10
    VOLUME_STD_PERIOD = 10
    VOLUME_MA_SHORT = 5
    VOLUME_MA_LONG = 10
    AMOUNT_MA_PERIOD = 10
    AMOUNT_STD_PERIOD = 10
    MFI_PERIOD = 28
    VR_PERIOD = 20
    VROC_PERIOD = 24
    VRSI_PERIOD = 15
    VMACD_FAST = 15
    VMACD_SLOW = 45
    VMACD_SIGNAL = 12
    ADOSC_FAST = 7
    ADOSC_SLOW = 20
    
    # 摆动指标参数
    KDJ_N = 21
    WILLR_PERIOD = 28
    BIAS_PERIOD = 24
    PSY_PERIOD = 12
    AR_BR_PERIOD = 20
    CR_PERIOD = 20
    
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
        print(f"默认模型: {cls.model.DEFAULT_MODELS}")
        print(f"XGBoost树数量: {cls.model.XGBOOST_PARAMS['n_estimators']}")
        print(f"学习率: {cls.model.XGBOOST_PARAMS['learning_rate']}")
        
        print("\n[训练配置]")
        print(f"训练集比例: {cls.training.TRAIN_TEST_SPLIT}")
        print(f"预测天数: {cls.training.FUTURE_DAYS}")
        print(f"路径预测: {'开启' if cls.training.USE_PATH_BASED_LABEL else '关闭'}")
        print(f"止盈/止损ATR倍数: {cls.training.ATR_TP_MULTIPLIER}x/{cls.training.ATR_SL_MULTIPLIER}x")
        
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
