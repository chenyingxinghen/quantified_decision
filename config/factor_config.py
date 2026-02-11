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
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 20,
    }
    
    # LightGBM配置
    LIGHTGBM_PARAMS = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    # Random Forest配置
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    # 默认使用的模型类型
    DEFAULT_MODELS = ['xgboost', 'lightgbm']
    
    @classmethod
    def get_model_params(cls, model_type: str) -> Dict[str, Any]:
        """获取指定模型的参数"""
        params_map = {
            'xgboost': cls.XGBOOST_PARAMS,
            'lightgbm': cls.LIGHTGBM_PARAMS,
            'random_forest': cls.RANDOM_FOREST_PARAMS,
        }
        return params_map.get(model_type, {}).copy()


# ============================================================================
# 2. 训练参数配置
# ============================================================================

class TrainingConfig:
    """训练参数配置"""
    # 模型训练任务类型
    TASK_TYPE = 'classification'  # 'classification', 'regression', 'ranking'
    # 收益率加权训练分类, TASK_TYPE = 'classification'
    USE_WEIGHT = False

    YEARS_FOR_TRAINING=3
    STOCK_NUM = 500
    # 数据集划分
    TRAIN_TEST_SPLIT = 0.8
    # 预测天数
    FUTURE_DAYS = 5
    # 分类阈值（涨幅 > 2%）
    RETURN_THRESHOLD = 0.02
    # 期望收益率阈值（回归任务）
    EXPECTED_RETURN_THRESHOLD = 0.01
    # 交易信号阈值
    SIGNAL_THRESHOLD = 0.5  # 调整为更通用的0.5
    
    # 缓存目录
    CACHE_DIR = 'data/factors_cache'
    # 模型保存目录
    SAVE_DIR = 'models'
    
    # 早停
    EARLY_STOPPING = True
    EARLY_STOPPING_ROUNDS = 20


# ============================================================================
# 3. 因子计算参数配置
# ============================================================================

class FactorConfig:
    """因子计算参数配置"""
    
    # 动量因子参数
    RSI_PERIOD = 21
    ROC_PERIOD = 30
    MTM_PERIOD = 30
    CMO_PERIOD = 28
    STOCHRSI_PERIOD = 9
    RVI_PERIOD = 20
    
    # 趋势因子参数
    MACD_FAST = 15
    MACD_SLOW = 20
    MACD_SIGNAL = 5
    ADX_PERIOD = 14
    DMI_PERIOD = 21
    AROON_PERIOD = 30
    TRIX_PERIOD = 15
    
    # 均线参数
    MA_RATIO_PERIOD = 60
    MA_SLOPE_PERIOD = 10
    
    # 波动率因子参数
    ATR_PERIOD = 20
    NATR_PERIOD = 21
    BB_PERIOD = 20
    BB_STD = 1.5
    CCI_PERIOD = 28
    ULCER_PERIOD = 7
    PRICE_VAR_PERIOD = 10
    
    # 成交量因子参数
    VOLUME_MA_PERIOD = 10
    VOLUME_STD_PERIOD = 10
    VOLUME_MA_SHORT = 5
    VOLUME_MA_LONG = 10
    AMOUNT_MA_PERIOD = 10
    AMOUNT_STD_PERIOD = 10
    MFI_PERIOD = 21
    VR_PERIOD = 26
    VROC_PERIOD = 18
    VRSI_PERIOD = 12
    VMACD_FAST = 6
    VMACD_SLOW = 30
    VMACD_SIGNAL = 12
    ADOSC_FAST = 3
    ADOSC_SLOW = 10
    
    # 摆动指标参数
    KDJ_N = 21
    WILLR_PERIOD = 14
    BIAS_PERIOD = 24
    PSY_PERIOD = 24
    AR_BR_PERIOD = 26
    CR_PERIOD = 26
    
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
    N_FEATURES_TO_SELECT = 40  # 增加特征数
    
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
        print(f"收益率阈值: {cls.training.RETURN_THRESHOLD}")
        
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
