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
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8, 

        'min_child_weight': 1,       # 增加权重要求，防止过拟合
        'gamma': 0.15,    
        'reg_alpha': 7,            
        'reg_lambda': 17,             
        'objective': 'reg:squarederror', 
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
        'learning_rate': 0.03,
        
        'min_child_weight': 1,
        'min_gain_to_split': 0.15,
        'reg_alpha': 7,
        'reg_lambda': 17,
        'subsample': 0.8,
        'colsample_bytree': 0.8,

        'label_gain': [float(i**1.5+1) for i in range(100)], 
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 512,

        'ndcg_eval_at': [i for i in range(1, 5)],  # 评估 ndcg@5'
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
    
    # 退市临近惩罚：对距退市日 N 个自然日以内的样本，将标签压至极低值
    # 模型推理时不依赖此特征，而是从量价/ST等共性模式中泛化学习
    DELIST_PENALTY_DAYS = 30     # 退市前多少天内的样本触发惩罚
    DELIST_PENALTY_SCORE = 0.01  # 惩罚后的标签值（远低于正常均值 ~1.0）
    
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
    RSI_PERIOD = 9            # 短线常用7~14，取9平衡灵敏度与稳定性
    ROC_PERIOD = 10           # 10日变动率捕捉短期加速
    MTM_PERIOD = 8            # 8日动量适应快速转折
    CMO_PERIOD = 14           # 钱德指标缩短至14日
    STOCHRSI_PERIOD = 12      # 随机RSI缩短至12
    RVI_PERIOD = 7            # 相对活力指数更短以快速确认方向
    
    # ========== 趋势因子参数 ==========
    MACD_FAST = 6             # 快线缩短提高交叉频率
    MACD_SLOW = 13            # 慢线大幅缩短
    MACD_SIGNAL = 5           # 信号线同步缩短
    ADX_PERIOD = 14           # 14日ADX识别短期趋势强度
    DMI_PERIOD = 14           # 与ADX保持一致
    AROON_PERIOD = 14         # 阿隆周期缩短，更快捕捉新高低点
    TRIX_PERIOD = 12          # 三重指数均线缩短至12
    
    # ========== 均线参数 ==========
    MA_RATIO_PERIOD = 20      # 使用20日均线衡量短期偏离
    MA_SLOPE_PERIOD = 8       # 8日均线斜率判断短期方向
    
    # ========== 波动率因子参数 ==========
    ATR_PERIOD = 10           # 10日ATR快速反映近期波幅
    NATR_PERIOD = 10          # 归一化ATR同周期
    BB_PERIOD = 20            # 布林带中轨20日均线，短线标准
    BB_STD = 1.5              # 1.5倍标准差带略宽过滤噪音
    CCI_PERIOD = 14           # 14日CCI捕捉短期超买超卖
    ULCER_PERIOD = 14         # 溃疡指数缩短
    PRICE_VAR_PERIOD = 10     # 价格方差窗口缩短
    
    # ========== 成交量因子参数 ==========
    VOLUME_MA_PERIOD = 5      # 短线维持5日均量
    VOLUME_STD_PERIOD = 8     # 略微缩短量标准差周期
    VOLUME_MA_SHORT = 3       # 快量线改为3日
    VOLUME_MA_LONG = 8        # 慢量线改为8日
    AMOUNT_MA_PERIOD = 5      # 成交额均线保持
    AMOUNT_STD_PERIOD = 8     # 成交额标准差缩短
    MFI_PERIOD = 14           # 资金流向指标缩短至14
    VR_PERIOD = 12            # 量比率缩短至12
    VROC_PERIOD = 10          # 量变动速率缩短至10
    VRSI_PERIOD = 9           # 量RSI缩短至9
    VMACD_FAST = 6            # 量MACD参数与价格MACD近似
    VMACD_SLOW = 13           
    VMACD_SIGNAL = 5          
    ADOSC_FAST = 2            # 佳庆振荡器快线缩短
    ADOSC_SLOW = 5            # 慢线缩短，信号更灵敏
    
    # ========== 摆动指标参数 ==========
    KDJ_N = 9                 # KDJ周期常用9
    WILLR_PERIOD = 14         # 威廉指标14日标准短线
    BIAS_PERIOD = 8           # 乖离率缩短到8日
    PSY_PERIOD = 12           # 心理线12日
    AR_BR_PERIOD = 13         # 人气意愿指标缩短至13
    CR_PERIOD = 13            # 中间意愿指标一致
    
    # ========== K线形态参数 ==========
    BODY_SIZE_THRESHOLD_LARGE = 0.012   # 短线中等波动即可视为大实体
    BODY_SIZE_THRESHOLD_SMALL = 0.0025  # 微调小实体识别精度
    HAMMER_LOWER_SHADOW_RATIO = 2.0     # 锤子线/上吊线下影线与实体的最小倍数
    HAMMER_UPPER_SHADOW_RATIO = 0.15     # 锤子线/上吊线上影线与实体的最大倍数
    SHOOTING_STAR_UPPER_RATIO = 2.0     # 射击之星/倒锤线上影线与实体的最小倍数
    SHOOTING_STAR_LOWER_RATIO = 0.15     # 射击之星/倒锤线下影线与实体的最大倍数
    DOJI_THRESHOLD = 0.003              # 十字星实体阈值（相对价格比例）
    MARUBOZU_SHADOW_RATIO = 0.002       # 光头光脚影线阈值（相对价格比例）
    MARUBOZU_MIN_BODY_RATIO = 0.015     # 光头光脚最小实体比例
    SPINNING_TOP_BODY_RATIO = 0.1       # 纺锤线实体/全幅最大比例
    SPINNING_TOP_SHADOW_SYMMETRY = 0.3  # 纺锤线上下影线对称性阈值
    ENGULFING_SIGNIFICANCE = 0.003      # 吞没形态显著性（超出前根实体的比例）
    STAR_SECOND_BODY_RATIO = 0.15        # 晨星/暮星第二根实体相对第一根的最大比例
    HARAMI_BODY_RATIO = 2.0             # 孕线外包实体相对内包实体的最小倍数
    CONTEXT_WINDOW = 20                # 上下文计算滚动窗口（天）
    CONTEXT_SIDEWAYS_MA_DEVIATION = 0.05  # 横盘判断：收盘价偏离均线的最大比例
    CONTEXT_SIDEWAYS_RANGE_PCT = 0.10   # 横盘判断：区间波动幅度最大比例
    # 价格位置阈值（0=低位, 1=高位）
    PRICE_POS_LOW = 0.25                 # 低位阈值（锤子线、晨星等看涨形态）
    PRICE_POS_HIGH = 0.75               # 高位阈值（射击之星、上吊线等看跌形态）
    PRICE_POS_LOW_STRICT = 0.25         # 严格低位阈值（倒锤线）
    PRICE_POS_HIGH_STRICT = 0.75         # 严格高位阈值（暮星）
    PRICE_POS_LOW_ENGULF = 0.25          # 吞没/刺穿线低位阈值
    PRICE_POS_HIGH_ENGULF = 0.8         # 吞没/乌云盖顶高位阈值
    PRICE_POS_SOLDIERS_CROWS = 0.25      # 三白兵/三乌鸦位置阈值


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

