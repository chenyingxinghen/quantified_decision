"""
因子模型配置文件

包含：
1. 模型超参数配置
2. 训练参数配置
3. 因子计算参数配置
4. 优化参数配置
"""

from typing import Dict, Any
from config import data_config
from config.jydb_config import JYDB_ENABLED, JYDB_FEATURE_DB_PATH
import os

n_bins = 15

# ============================================================================
# 1. 模型超参数配置
# ============================================================================

class ModelConfig:
    """模型超参数配置"""

    # ── LightGBM Ranking 配置 ─────────────────────────────────────────────
    LIGHTGBM_PARAMS: Dict[str, Any] = {
        'n_estimators': 2000,
        'num_leaves': 10,          
        'learning_rate': 0.02,    

        'min_child_weight': 1,
        'min_gain_to_split': 0.01, # 最小分裂增益，剪掉无意义的分裂
        'reg_alpha': 0.01,          # L1 正则，促进稀疏性
        'reg_lambda': 0.01,         # L2 正则，平滑权重

        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'subsample_freq': 1,

        # Ranking 专属配置
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [5],
        'lambdarank_truncation_level': 50,
        'label_gain': [i//3*i**1.5 if i>2 else i for i in range(n_bins)],

        'early_stopping_rounds': 200,
        'n_jobs': -1,  # 使用所有CPU核心
        'verbosity': -1,
    }


    XGBOOST_PARAMS: Dict[str, Any] = {
        'n_estimators': 2000,
        'max_depth': 3,
        'learning_rate': 0.02,

        'subsample': 0.8,
        'colsample_bytree': 0.5,

        'min_child_weight': 2.0,
        'gamma': 0.01,              # 最小分裂损失，剪掉无意义的分裂
        'reg_alpha': 0.2,          # L1 正则
        'reg_lambda': 0.2,         # L2 正则

        # Ranking 专属配置
        'eval_metric': 'ndcg',
        'ndcg_exp_gain': False,  

        'n_jobs': -1,  # 使用所有CPU核心
        'early_stopping_rounds': 200,
        'verbosity': 1,
    }



    # ── GPU 专用配置增量（XGBoost，当 USE_GPU=True 时叠加）───────────────
    GPU_PARAMS_XGB: Dict[str, Any] = {
        'tree_method': 'hist',   # XGBoost 2.0+ 推荐 hist + device=cuda
        'device': 'cuda',
        'n_jobs': 1,             # CPU 核心数：设为物理核数一半，保证 CPU 预处理不拖 GPU 后腿
    }

    # ── 统一接口 ──────────────────────────────────────────────────────────
    @classmethod
    def get_model_params(cls, model_type: str, task: str = None) -> Dict[str, Any]:
        """获取指定模型的超参数，根据任务类型动态设置目标"""
        if task is None:
            task = TrainingConfig.TASK
        
        params_map = {
            'xgboost': cls.XGBOOST_PARAMS.copy(),
            'lightgbm': cls.LIGHTGBM_PARAMS.copy(),
        }
        params = params_map.get(model_type, {})
        
        # 根据任务类型设置目标
        if task == 'hybrid':
            if model_type == 'xgboost':
                params['objective'] = 'reg:squarederror'
            elif model_type == 'lightgbm':
                params['objective'] = 'lambdarank'
        elif task == 'ranking':
            if model_type == 'xgboost':
                params['objective'] = 'rank:ndcg'
            elif model_type == 'lightgbm':
                params['objective'] = 'lambdarank'
        elif task == 'regression':
            if model_type == 'xgboost':
                params['objective'] = 'reg:squarederror'
            elif model_type == 'lightgbm':
                params['objective'] = 'regression'
        
        if model_type == 'xgboost' and not os.getenv('QD_DISABLE_GPU'):
            params.update(cls.GPU_PARAMS_XGB)
        return params

    @classmethod
    def get_n_bins(cls) -> int:
        """
        返回两模型统一使用的标签档位数（N_BINS）。

        设计原则：
        - LightGBM 以 label_gain 长度为准（决定 lambdarank 的增益曲线）。
        - XGBoost 复用相同档位数，保证两者离散标签的语义一致。
        """
        return n_bins


# ============================================================================
# 2. 训练参数配置
# ============================================================================

class TrainingConfig:
    """训练参数配置"""

    # ── 模型 ──────────────────────────────────────────────────────────────
    MODEL_TYPES          = ['lightgbm','xgboost']
    TASK                 = 'ranking' 

    # ── 数据范围 ───────────────────────────────────────────────────────────
    YEARS                = data_config.HISTORY_YEARS
    YEARS_FOR_TRAINING   = 6         # 训练数据年数
    YEARS_FOR_BACKTEST   = 0.5         # 回测数据年数
    STOCK_NUM            = 6000      # 参与训练的股票数量上限
    SHORT_PREDICTION     = False
    FUTURE_DAYS          = 7 if SHORT_PREDICTION else 15         # 预测未来 N 个交易日
    MULTI_OBJECTIVE_RETURN_HORIZONS = (5, 20, 60)
    MULTI_OBJECTIVE_RISK_HORIZON = 20
    # 正交化相关开关（标签构造层）
    #  True  → 构造「非重叠前向收益腿」y_ret_leg_1_5d / _6_20d / _21_60d，
    #          该组目标窗口互不相交、按构造正交，是无泄露的正交目标基。
    MULTI_OBJECTIVE_ORTHOGONAL_LEGS = True
    #  True  → build_universe 后对目标做横截面 Gram-Schmidt 残差化（orth_* 列），
    #          使任意目标集合在同日截面内两两正交；仅用当日截面，无泄露。
    #  注意：开启后训练目标应改用 orth_* 列（需重新训练，旧模型不再对齐）。
    MULTI_OBJECTIVE_ORTHOGONALIZE = False
    # Sharpe 类标签是否使用「与收益同长」的回撤分母（修复跨期泄露，默认开启）。
    MULTI_OBJECTIVE_MATCHING_RISK_SHARPE = True
    MULTI_OBJECTIVE_WEIGHTS = {
        'y_ret_5d': 0.20,
        'y_ret_20d': 0.25,
        'y_ret_60d': 0.20,
        'y_mdd_20d': 0.15,
        'y_downvol_20d': 0.08,
        'y_illiq_20d': 0.07,
        'y_tradable_20d': 0.05,
    }


    # ── 数据集划分 ─────────────────────────────────────────────────────────
    TRAIN_TEST_SPLIT     = 0.8       # 训练集占比

    # ── 因子与基本面 ───────────────────────────────────────────────────────
    INCLUDE_FUNDAMENTALS = True      # 是否包含基本面因子
    # 聚源清洗后的 PIT 特征。数据库不存在时自动降级为空特征，保持旧流程兼容。
    INCLUDE_JYDB_FEATURES = JYDB_ENABLED or os.path.exists(JYDB_FEATURE_DB_PATH)
    INCLUDE_CANDLE_PATTERN = False

    # 标签变换：回归与 XGBoost ranking 共用连续标签变换；LightGBM ranking 由 label_gain 控制
    LABEL_WEIGHTED_FOR_XGB= True
    LABEL_WEIGHT_EXPONENT=1.2
    
    UPSIDE_WEIGHT        = 1.0       
    DOWNSIDE_WEIGHT      = 1.0       
    FINAL_RETURN_WEIGHT  = 1.0       




    # ── 路径形态奖惩 (f_high_idx vs f_low_idx) ───────────────────────────
    PATH_BONUS           = 0.15      # 先涨后跌路径奖励幅度（+15%）
    PATH_PENALTY         = 0.10      # 先跌后涨路径惩罚幅度（-10%）
    

    WEIGHT_EXPONENT      = 2         # 适度头部加权，让模型更关注真正的强势股信号
    USE_SAMPLE_WEIGHT    = False      # 开启样本加权，引导模型关注高质量预测目标

    # ST 股票处理
    ST_LABEL_SCORE       = -50         # ST 样本原始分上限（0=中性）
    ST_WEIGHT_FACTOR     = 0.1       # ST 样本权重降低因子
    
    # 退市预警处理
    DELIST_PENALTY_DAYS  = 60      
    DELIST_PENALTY_SCORE = -100      # 退市样本直接给最低分
    UNBUYABLE_HANDLING   = 'remove'  
    UNBUYABLE_PENALTY_SCORE = 0.0  # 次日无法成交的样本强制进入当日最低标签档
    
        

    # ── 基础设施与计算性能 (Infrastructure) ────────────────────────────────
    USE_GPU              = True          # 是否启用 GPU 加速（XGBoost/LightGBM）
    MEMORY_EFFICIENT     = True          # 是否启用分批训练/流式加载
    GPU_BATCH_SIZE       = 1_000_000     # GPU 批次大小
    N_JOBS_FACTOR_CALC   = 12             # 横截面归一化线程数；过高会增加峰值内存与调度开销

    # ── 因子归一化 (Feature Normalization) ────────────────────────────────
    # 在进行横截面排名时，跳过这些特定类型的因子
    FEATURE_RANK_SKIP_LIST = {
        # 1. 全市场/宏观因子 (所有股票值都一样，排名无意义)
        'up_ratio', 'strong_up_ratio', 'down_ratio', 'limit_up_ratio', 
        'limit_down_ratio', 'mean_return', 'total_volume', 'adv_vol_ratio', 
        'breadth_ma20',
        
        # 2. 状态/类型因子 (离散值)
        'market_type', 'is_limit_up', 'is_suspended', 'is_st',
        
        # 3. K线形态 (0/1 二元标记)
        'white_candle', 'black_candle', 'doji', 'hammer', 'hanging_man',
        'shooting_star', 'inverted_hammer', 'marubozu', 'spinning_top',
        'bullish_engulfing', 'bearish_engulfing', 'piercing_line',
        'dark_cloud_cover', 'morning_star', 'evening_star', 'harami',
        'three_white_soldiers', 'three_black_crows',
        
        # 4. 行业/板块 (One-hot 编码)
        # 逻辑：以 industry_ 或 sector_ 开头，且不以 _encoded 结尾的（通常是原始字符串或ID）
    }

    # ── 特征工程衍生过滤 (Feature Engineering Transformation Filtering) ──────
    # 在自动生成比率、乘积、差值等衍生特征时，严禁使用以下类型的原始特征作为分母或交互项
    # 避免产生无意义的"除以状态位"导致的虚假重要性（如 adv_vol_ratio_div_is_suspended）
    FEATURE_TRANSFORM_EXCLUDE_LIST = {
        'market_type', 'is_limit_up', 'is_suspended', 'is_st',
        'white_candle', 'black_candle', 'doji', 'hammer', 'hanging_man',
        'shooting_star', 'inverted_hammer', 'marubozu', 'spinning_top',
        'bullish_engulfing', 'bearish_engulfing', 'piercing_line',
        'dark_cloud_cover', 'morning_star', 'evening_star', 'harami',
        'three_white_soldiers', 'three_black_crows',
        'up_ratio', 'strong_up_ratio', 'down_ratio', 'limit_up_ratio',
        'limit_down_ratio', 'mean_return', 'total_volume', 'adv_vol_ratio',
        'breadth_ma20', 'days_to_delist',
        
        # 4. 绝对规模因子 (防止在特征工程中产生"因子 x 规模"导致的严重 Size Bias)
        'MBRevenue', 'totalShare', 'liqaShare', 'netProfit', 'market_cap', 
        'totalAssets', 'totalLiabilities', 'epsTTM', 'cashFlow', 'log_mkt_cap','liabilityToAsset','aroon'
    }

    @staticmethod
    def should_skip_transform(col: str) -> bool:
        """判定该因子是否应跳过特征工程变换 (如比率、交互项等)"""
        if col in TrainingConfig.FEATURE_TRANSFORM_EXCLUDE_LIST:
            return True
        col_l = col.lower()
        # 外部结构化特征在训练前统一做横截面 Rank；对其再做 log/sqrt 等
        # 单调变换不会增加排序信息，只会扩大稀疏事件列和类别列的候选空间。
        if col_l.startswith('jy_'):
            return True
        # 排除所有状态位、K线形态、宏观指标
        if col_l.startswith(('industry_', 'jy_industry_', 'sector_', 'is_', 'days_to_', 'market_')):
            return True
        # 排除 FEATURE_TRANSFORM_EXCLUDE_LIST 中以前缀方式匹配的条目（如 'aroon' 匹配 'aroon_up'）
        for excl in TrainingConfig.FEATURE_TRANSFORM_EXCLUDE_LIST:
            if col_l.startswith(excl.lower() + '_'):
                return True
        return False

    @staticmethod
    def should_skip_rank(col: str) -> bool:
        """判定该因子是否应跳过横截面排名"""
        if col in TrainingConfig.FEATURE_RANK_SKIP_LIST:
            return True
        col_l = col.lower()
        # 行业、板块分类、退市天数、二元标记、市场/指标前缀
        if col_l.startswith(('industry_', 'jy_industry_', 'sector_', 'is_', 'days_to_', 'market_')):
            # 如果是已经 label encoding 过的行业因子，可以参与排名（保持分布一致）
            if col_l.endswith('_encoded'):
                return False
            return True
        return False

    # ── 路径 ───────────────────────────────────────────────────────────────
    # 云端适配：
    #  - 因子缓存需在可写目录生成（平台挂载的数据为只读），云端用 GEMINI_CACHE_DIR 指向 $GEMINI_DATAOUT 下目录；
    #  - 模型产物需落盘到 $GEMINI_DATAOUT 才能持久化，云端用 GEMINI_SAVE_DIR 覆盖。
    # 本地运行不设置这些变量时回退到项目内默认相对路径（相对 cwd=项目根）。
    CACHE_DIR = os.getenv("GEMINI_CACHE_DIR", "database/system_data/factors_cache")
    SAVE_DIR  = os.getenv("GEMINI_SAVE_DIR", "models")

    # ── 训练股票池过滤（与 strategy_config 中的选股条件对齐）──────────────
    # 开启后，训练数据只包含满足策略选股条件的股票，使训练分布与推理分布一致。
    # 关闭后，使用全市场股票训练，模型具备更强的跨股票池泛化能力（推荐）。
    # 注意：开启此选项会显著减少训练样本量，可能导致过拟合。
    FILTER_BY_STRATEGY   = False

    # 当 FILTER_BY_STRATEGY=True 时生效的具体过滤条件
    # None 表示不限制该条件，与 strategy_config 中的语义一致
    TRAIN_FILTER_MARKETS  = None   # None = 使用 strategy_config.SELECTOR_MARKETS
    TRAIN_FILTER_MAX_PRICE = None  # None = 使用 strategy_config.MAX_PRICE
    TRAIN_FILTER_MIN_PRICE = None  # None = 使用 strategy_config.MIN_PRICE
    TRAIN_FILTER_INCLUDE_ST = None # None = 使用 strategy_config.INCLUDE_ST


# ============================================================================
# 3. 因子计算参数配置
# ============================================================================

class FactorConfig:
    """
    因子计算参数配置
    定义各类技术指标与 K 线形态的计算窗口与阈值
    """

    # ========== 动量因子参数 ==========
    RSI_PERIOD = 21
    ROC_PERIOD = 30
    MTM_PERIOD = 10
    CMO_PERIOD = 21
    STOCHRSI_PERIOD = 20
    RVI_PERIOD = 14           

    # ========== 趋势因子参数 ==========
    MACD_FAST = 20
    MACD_SLOW = 60
    MACD_SIGNAL = 2
    ADX_PERIOD = 20
    DMI_PERIOD = 20
    AROON_PERIOD = 50
    TRIX_PERIOD = 30

    # ========== 均线参数 ==========
    MA_RATIO_PERIOD = 50
    MA_SLOPE_PERIOD = 14

    # ========== 波动率因子参数 ==========
    ATR_PERIOD = 14
    NATR_PERIOD = 28
    BB_PERIOD = 50
    BB_STD = 1.5               # 1.5倍标准差带略宽过滤噪音
    CCI_PERIOD = 20
    ULCER_PERIOD = 7
    PRICE_VAR_PERIOD = 30

    # ========== 成交量因子参数 ==========
    VOLUME_MA_PERIOD = 14
    VOLUME_STD_PERIOD = 30
    VOLUME_MA_SHORT = 5        # 快量线
    VOLUME_MA_LONG = 14        # 慢量线
    AMOUNT_MA_PERIOD = 10
    AMOUNT_STD_PERIOD = 30
    MFI_PERIOD = 15
    VR_PERIOD = 30
    VROC_PERIOD = 18
    VRSI_PERIOD = 21
    VMACD_FAST = 18
    VMACD_SLOW = 25
    VMACD_SIGNAL = 20
    ADOSC_FAST = 2
    ADOSC_SLOW = 7

    # ========== 摆动指标参数 ==========
    KDJ_N = 14
    WILLR_PERIOD = 20
    BIAS_PERIOD = 18
    PSY_PERIOD = 30
    AR_BR_PERIOD = 26
    CR_PERIOD = 52

    # ========== K线形态参数 ==========
    BODY_SIZE_THRESHOLD_LARGE = 0.012   # 短线中等波动即可视为大实体
    BODY_SIZE_THRESHOLD_SMALL = 0.0025  # 微调小实体识别精度
    HAMMER_LOWER_SHADOW_RATIO = 2.0     # 锤子线/上吊线下影线与实体的最小倍数
    HAMMER_UPPER_SHADOW_RATIO = 0.15    # 锤子线/上吊线上影线与实体的最大倍数
    SHOOTING_STAR_UPPER_RATIO = 2.0     # 射击之星/倒锤线上影线与实体的最小倍数
    SHOOTING_STAR_LOWER_RATIO = 0.15    # 射击之星/倒锤线下影线与实体的最大倍数
    DOJI_THRESHOLD = 0.003              # 十字星实体阈值（相对价格比例）
    MARUBOZU_SHADOW_RATIO = 0.002       # 光头光脚影线阈值（相对价格比例）
    MARUBOZU_MIN_BODY_RATIO = 0.015     # 光头光脚最小实体比例
    SPINNING_TOP_BODY_RATIO = 0.1       # 纺锤线实体/全幅最大比例
    SPINNING_TOP_SHADOW_SYMMETRY = 0.3  # 纺锤线上下影线对称性阈值
    ENGULFING_SIGNIFICANCE = 0.003      # 吞没形态显著性（超出前根实体的比例）
    STAR_SECOND_BODY_RATIO = 0.15       # 晨星/暮星第二根实体相对第一根的最大比例
    HARAMI_BODY_RATIO = 2.0             # 孕线外包实体相对内包实体的最小倍数
    CONTEXT_WINDOW = 20                 # 上下文计算滚动窗口（天）
    CONTEXT_SIDEWAYS_MA_DEVIATION = 0.05   # 横盘判断：收盘价偏离均线的最大比例
    CONTEXT_SIDEWAYS_RANGE_PCT = 0.10      # 横盘判断：区间波动幅度最大比例
    # 价格位置阈值（0=低位, 1=高位）
    PRICE_POS_LOW = 0.25                # 低位阈值（锤子线、晨星等看涨形态）
    PRICE_POS_HIGH = 0.75               # 高位阈值（射击之星、上吊线等看跌形态）
    PRICE_POS_LOW_STRICT = 0.25         # 严格低位阈值（倒锤线）
    PRICE_POS_HIGH_STRICT = 0.75        # 严格高位阈值（暮星）
    PRICE_POS_LOW_ENGULF = 0.25         # 吞没/刺穿线低位阈值
    PRICE_POS_HIGH_ENGULF = 0.8         # 吞没/乌云盖顶高位阈值
    PRICE_POS_SOLDIERS_CROWS = 0.25     # 三白兵/三乌鸦位置阈值


# ============================================================================
# 4. 自动化优化与特征工程配置
# ============================================================================

class OptimizationConfig:
    """
    优化参数配置
    定义特征选择、超参数搜索与集成学习的策略
    """

    # 特征选择方法
    FEATURE_SELECTION_METHOD = 'hybrid'  # 'importance', 'correlation', 'mutual_info', 'rfe', 'hybrid'
    N_FEATURES_TO_SELECT = 400  # 候选上限；最终数量仍受覆盖率、常量和相关性过滤约束

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
