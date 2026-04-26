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
        'max_depth': 7,              # 增加深度以改善预测区分度
        'learning_rate': 0.03, 
      
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'colsample_bylevel': 0.8, 

        'min_child_weight': 1,       # 增加权重要求，防止过拟合
        'gamma': 0.1,    
        'reg_alpha': 7,            
        'reg_lambda': 17,             
        'objective': 'reg:logistic', 
        'eval_metric': 'auc',       
        'n_jobs': 15,
        'early_stopping_rounds': 50,
        'verbosity': 0,              # 打印训练过程
    }
    
    # LightGBM配置
    LIGHTGBM_PARAMS = {
        'n_estimators': 3000,
        'max_depth': 7,
        'num_leaves': 127,
        'learning_rate': 0.05,
        
        'min_child_weight': 1,
        'min_gain_to_split': 0.1,
        'reg_alpha': 3,
        'reg_lambda': 17,
        'subsample': 0.8,
        'colsample_bytree': 0.8,

        'label_gain': [float(i**2+1) for i in range(100)], 
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 100,

        'ndcg_eval_at': [i for i in range(1, 5)],  # 评估 ndcg@100'
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
    MODEL_TYPES = ['xgboost','lightgbm']
    SAMPLE_EVAL = True    # 是否使用随机采样评估

    INCLUDE_FUNDAMENTALS = True  # 是否包含基本面因子
    PUNISH_UNBUYABLE = True      # 涨停板、停牌样本惩罚 (兼容旧逻辑)
    ST_WEIGHT_FACTOR = 0.5       # ST 股票样本权重降低因子 (0.5 表示权重减半)
    UNBUYABLE_HANDLING = 'remove' # 'remove' (推荐，剔除样本) 或 'punish' (惩罚，软标签设为 0.05)
    
    # XGB标签构造参数 LGB样本权重构造参数
    LABEL_TARGET_SCALE = 2
    LABEL_LAMBDA = 1           # 损失厌恶系数 (惩罚回撤)
    LABEL_PATH_PUNISH = 0.5      # 路径保护惩罚系数 (惩罚先跌后涨)
    LABEL_TIME_BONUS = 0.8       # 时间奖励系数 (奖励高点早的票)
    
    USE_GPU = True               # 是否启用 GPU 加速
    MEMORY_EFFICIENT = True      # 是否启用内存优化模式 (针对大规模数据集)
    GPU_BATCH_SIZE = 100000     # GPU 分批训练的批大小 (增加此值可提高 GPU 利用率)

    YEARS=baostock_config.HISTORY_YEARS

    YEARS_FOR_BACKTEST=1         # 回测年数
    YEARS_FOR_TRAINING=16         # 训练年数
    STOCK_NUM = 6000             # 股票数量
    # 数据集划分
    TRAIN_TEST_SPLIT = 0.8
    

    # 预测天数 (用于分类、回归和排序任务)
    FUTURE_DAYS = 7
    

    # 缓存目录
    CACHE_DIR = 'database/system_data/factors_cache'
    # 模型保存目录
    SAVE_DIR = 'models'
    


# ============================================================================
# 3. 因子计算参数配置
# ============================================================================

class FactorConfig:
    """因子计算参数配置（优化为短线策略，适合1-5日持仓）"""
    
    # ========== 动量因子参数 ==========
    RSI_PERIOD = 9            # 原42，短线常用7~14，取9平衡灵敏度与稳定性
    ROC_PERIOD = 10           # 原60，10日变动率捕捉短期加速
    MTM_PERIOD = 8            # 原40，8日动量适应快速转折
    CMO_PERIOD = 14           # 原42，钱德指标缩短至14日
    STOCHRSI_PERIOD = 12      # 原28，随机RSI缩短至12
    RVI_PERIOD = 7            # 原14，相对活力指数更短以快速确认方向
    
    # ========== 趋势因子参数 ==========
    MACD_FAST = 6             # 原25，快线缩短提高交叉频率
    MACD_SLOW = 13            # 原90，慢线大幅缩短
    MACD_SIGNAL = 5           # 原20，信号线同步缩短
    ADX_PERIOD = 14           # 原35，14日ADX识别短期趋势强度
    DMI_PERIOD = 14           # 原35，与ADX保持一致
    AROON_PERIOD = 14         # 原50，阿隆周期缩短，更快捕捉新高低点
    TRIX_PERIOD = 12          # 原30，三重指数均线缩短至12
    
    # ========== 均线参数 ==========
    MA_RATIO_PERIOD = 20      # 原120，使用20日均线衡量短期偏离
    MA_SLOPE_PERIOD = 8       # 原30，8日均线斜率判断短期方向
    
    # ========== 波动率因子参数 ==========
    ATR_PERIOD = 10           # 原30，10日ATR快速反映近期波幅
    NATR_PERIOD = 10          # 原28，归一化ATR同周期
    BB_PERIOD = 20            # 原100，布林带中轨20日均线，短线标准
    BB_STD = 1.5              # 保持不变，1.5倍标准差带略宽过滤噪音
    CCI_PERIOD = 14           # 原35，14日CCI捕捉短期超买超卖
    ULCER_PERIOD = 14         # 原35，溃疡指数缩短
    PRICE_VAR_PERIOD = 10     # 原30，价格方差窗口缩短
    
    # ========== 成交量因子参数 ==========
    VOLUME_MA_PERIOD = 5      # 原5，短线维持5日均量
    VOLUME_STD_PERIOD = 8     # 原10，略微缩短量标准差周期
    VOLUME_MA_SHORT = 3       # 原5，快量线改为3日
    VOLUME_MA_LONG = 8        # 原10，慢量线改为8日
    AMOUNT_MA_PERIOD = 5      # 原5，成交额均线保持
    AMOUNT_STD_PERIOD = 8     # 原10，成交额标准差缩短
    MFI_PERIOD = 14           # 原35，资金流向指标缩短至14
    VR_PERIOD = 12            # 原52，量比率缩短至12
    VROC_PERIOD = 10          # 原36，量变动速率缩短至10
    VRSI_PERIOD = 9           # 原21，量RSI缩短至9
    VMACD_FAST = 6            # 原25，量MACD参数与价格MACD近似
    VMACD_SLOW = 13           # 原60
    VMACD_SIGNAL = 5          # 原20
    ADOSC_FAST = 2            # 原3，佳庆振荡器快线缩短
    ADOSC_SLOW = 5            # 原7，慢线缩短，信号更灵敏
    
    # ========== 摆动指标参数 ==========
    KDJ_N = 9                 # 原28，KDJ周期常用9
    WILLR_PERIOD = 14         # 原35，威廉指标14日标准短线
    BIAS_PERIOD = 8           # 原36，乖离率缩短到8日
    PSY_PERIOD = 12           # 原30，心理线12日
    AR_BR_PERIOD = 13         # 原52，人气意愿指标缩短至13
    CR_PERIOD = 13            # 原52，中间意愿指标一致
    
    # ========== K线形态参数 ==========
    BODY_SIZE_THRESHOLD_LARGE = 0.012   # 原0.015，短线中等波动即可视为大实体
    BODY_SIZE_THRESHOLD_SMALL = 0.0025  # 原0.003，微调小实体识别精度
    HAMMER_LOWER_SHADOW_RATIO = 1.5     # 保持
    HAMMER_UPPER_SHADOW_RATIO = 0.5     # 保持


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
