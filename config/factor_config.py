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

n_bins = 15

# ============================================================================
# 1. 模型超参数配置
# ============================================================================

class ModelConfig:
    """模型超参数配置"""

    # ── LightGBM Ranking 配置 ─────────────────────────────────────────────
    LIGHTGBM_PARAMS: Dict[str, Any] = {
        'n_estimators': 1000,
        'num_leaves': 7,
        'learning_rate': 0.04,

        'min_child_weight': 2.5,
        'min_gain_to_split': 0.01, # 最小分裂增益，剪掉无意义的分裂
        'reg_alpha': 2.5,          # L1 正则，促进稀疏性
        'reg_lambda': 2.5,         # L2 正则，平滑权重

        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'subsample_freq': 1,

        # Ranking 专属配置
        'objective': 'lambdarank',
        'metric': 'ndcg',
        # 早停 first_metric_only=True 只盯排序后 eval_at 的最小截断位。用 @20 而非 @5：
        # A股 top-5 截面噪声大，@20 对 top-k 选股更稳健、随迭代更单调，作早停主指标更可靠。
        'eval_at': [20,50],
        'lambdarank_truncation_level': 100,
        'label_gain': [round(i ** 1.5, 4) for i in range(n_bins)],

        # ── 早停开启 (200 轮)，这是冠军 (7/28) 的健康配置，勿再关 ──
        # 教训：曾有一次全量重训 lgb 在第 [1] 轮崩溃 (Unique=7, best_iter=1)，我误判为
        # "9M 大样本下 lambdarank 病态"并关掉早停强训 600 轮 —— 结果制造了过拟合
        # (train IC 0.11 / val IC 0.01)，回测 -9.14%，远差于冠军 +17.27%。
        # 真因经诊断另有其人：缓存路径特征泄漏 (raw turnover_rate/amount + is_suspended
        # 混入 X，见 train_ml_model _extract_stock_components_from_cache 的 keep_cols 修复)，
        # 把强信号的【原始未归一化列】和零方差状态位灌进模型，扭曲了训练与早停曲线。
        # 反证：冠军同为 9M/17y/5480，早停正常停在 39 棵、健康。特征泄漏修复后早停即恢复。
        # 弱信号场景 (单特征 IC 天花板 ≈0.07) 就该【浅模型+强正则+早停】，绝不关早停。
        'early_stopping_rounds': 200,
        'n_jobs': -1,  # 使用所有CPU核心
        'verbosity': -1,
    }


    XGBOOST_PARAMS: Dict[str, Any] = {
        'n_estimators': 1000,
        'max_depth': 3,
        'learning_rate': 0.04,

        'subsample': 0.8,
        'colsample_bytree': 0.8,

        'min_child_weight': 2.5,
        'gamma': 0.01,              # 最小分裂损失，剪掉无意义的分裂
        'reg_alpha': 2.5,          # L1 正则
        'reg_lambda': 2.5,         # L2 正则

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
        'n_jobs': 4,
    }

    # ── 统一接口 ──────────────────────────────────────────────────────────
    # 全局随机种子。None = 沿用各框架默认（xgb seed=0 / lgb 固定默认），
    # 与 2026-08-15 之前的全部历史训练逐位一致，所以默认值必须保持 None。
    # 树只有 subsample/colsample 抽样受它影响，敏感度远低于 NAM；设它的唯一
    # 用途是**配对多种子判定**（T092 量化过：单种子回测熊市 MDE≈64pp，
    # n=4 配对能压到≈13pp），由 train_model.py --seed 注入。
    MODEL_SEED: Any = None

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
                params['objective'] = getattr(
                    TrainingConfig, 'XGBOOST_RANKING_OBJECTIVE', 'rank:ndcg'
                )
            elif model_type == 'lightgbm':
                params['objective'] = 'lambdarank'
        elif task == 'regression':
            if model_type == 'xgboost':
                params['objective'] = 'reg:squarederror'
            elif model_type == 'lightgbm':
                params['objective'] = 'regression'
        
        if model_type == 'xgboost':
            params.update(cls.GPU_PARAMS_XGB)
        # 两个框架的 sklearn wrapper 都认 random_state（xgb 内部映射到 seed）。
        # 只在显式设置时注入，未设置时不写这个键 —— 保持历史训练的逐位可复现。
        if cls.MODEL_SEED is not None:
            params['random_state'] = int(cls.MODEL_SEED)
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
    YEARS                = baostock_config.HISTORY_YEARS
    YEARS_FOR_TRAINING   = 17         # 训练数据年数
    YEARS_FOR_BACKTEST   = 2         # 回测数据年数
    STOCK_NUM            = 6000      # 参与训练的股票数量上限
    SHORT_PREDICTION     = True
    FUTURE_DAYS          = 7 if SHORT_PREDICTION else 15         # 预测未来 N 个交易日


    # ── 数据集划分 ─────────────────────────────────────────────────────────
    TRAIN_TEST_SPLIT     = 0.8       # 训练集占比

    # ── 因子与基本面 ───────────────────────────────────────────────────────
    INCLUDE_FUNDAMENTALS = True      # 是否包含基本面因子
    INCLUDE_CANDLE_PATTERN = False

    # ── 市场 Regime 注入方式 ───────────────────────────────────────────────
    # True  : 沿用手工 {mkt}_regime_{stock} 乘积交互（feature_engineering.py），
    #         即 tag `baseline-xgb-7d` 的生产行为。因子 parquet 缓存中已含这些列，
    #         改动会使缓存与 feature_names 失配，故默认保持 True。
    # False : 关闭手工交互。市场状态改由 core/factors/regime_features.py 的
    #         多时间尺度矩阵 M 提供，交给 NAMGateModel 的 RegimeGate 端到端学习。
    #         NAM 实验线若复用已有缓存，可不改此开关，直接在装载后剔除 *_regime_* 列。
    ENABLE_MARKET_INTERACTION = True

    # 标签变换：回归与 XGBoost ranking 共用连续标签变换；LightGBM ranking 由 label_gain 控制
    LABEL_WEIGHTED_FOR_XGB= True
    # 2026-08 多窗口验证：0.8 在 2019/2021/2023 起始的三个验证阶段均提高 Rank IC，
    # 且避免 1.2 在近期数据上 best_iter=1~7、预测分数近乎退化的问题。
    # Top-5 超额收益并非每个窗口都占优，因此保留完整训练后的头部指标晋级门槛。
    LABEL_WEIGHT_EXPONENT=0.8
    XGBOOST_RANKING_OBJECTIVE = 'rank:ndcg'
    
    UPSIDE_WEIGHT        = 1.0
    DOWNSIDE_WEIGHT      = 1.0
    FINAL_RETURN_WEIGHT  = 1.0

    # 波动率加成系数：标签分数 = base * (1 + rel_atr * VOL_BOOSTER_COEF) * path_mult
    # 诊断结论：高波动股未来收益偏低（低波动异象），正系数会与最强 alpha 反向。
    # 0=关闭波动率加成（对照实验用），10.0=原始行为。
    # 对照实验结论(2026-07)：coef=10 的验证 Rank IC(0.050) 明显优于 coef=0(0.031)，
    # 即使高波动股单变量 IC 为负——vol_booster 增强了截面排序区分度，对 lambdarank 有益。保持 10.0。
    VOL_BOOSTER_COEF     = 10.0




    # ── 路径形态奖惩 (f_high_idx vs f_low_idx) ───────────────────────────
    PATH_BONUS           = 0.15      # 先涨后跌路径奖励幅度（+15%）
    PATH_PENALTY         = 0.10      # 先跌后涨路径惩罚幅度（-10%）
    

    WEIGHT_EXPONENT      = 2         # 适度头部加权，让模型更关注真正的强势股信号
    USE_SAMPLE_WEIGHT    = False      # 开启样本加权，引导模型关注高质量预测目标

    # 时间衰减权重：近期市场结构可能对验证期更有代表性。
    # 权重按交易日计算，同一截面内所有股票权重一致，适配 XGBoost ranking 的 per-group
    # 权重语义，也避免改变单日内部的股票相对重要性。
    # 2026-08-03 对照（500股/8年/300树）：
    # off/2y/4y/8y 的验证 Rank IC = 0.0465/0.0294/0.0345/0.0403。
    # 当前特征与标签下，历史样本提供了有效的跨周期正则，时间衰减反而降低泛化，默认关闭。
    USE_RECENCY_WEIGHT       = False
    RECENCY_HALF_LIFE_YEARS  = 4.0
    RECENCY_MIN_WEIGHT       = 0.10

    # Ranking query 权重：按每日原始标签的截面 IQR 对完整交易日加权。
    # 权重在 query 内严格一致；默认关闭，实验候选由脚本临时覆盖。
    QUERY_LABEL_DISPERSION_WEIGHT = 'off'  # off / high / low
    QUERY_LABEL_WEIGHT_MIN = 0.75
    QUERY_LABEL_WEIGHT_MAX = 1.25

    # Ranking query 权重的 ATR regime 版本：按每日平均 atr_rel 的跨日分位加权。
    QUERY_ATR_REGIME_WEIGHT = 'off'  # off / high / low
    QUERY_ATR_WEIGHT_MIN = 0.75
    QUERY_ATR_WEIGHT_MAX = 1.25

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
    N_JOBS_FACTOR_CALC   = 2             # 横截面归一化线程数；过高会增加峰值内存与调度开销

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
        # 排除所有状态位、K线形态、宏观指标
        if col_l.startswith(('industry_', 'sector_', 'is_', 'days_to_', 'mkt_', 'market_', 'index_', 'sentiment_', 'vix_')):
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
        if col_l.startswith(('industry_', 'sector_', 'is_', 'days_to_', 'mkt_', 'market_', 'index_', 'sentiment_', 'vix_')):
            # 如果是已经 label encoding 过的行业因子，可以参与排名（保持分布一致）
            if col_l.endswith('_encoded'):
                return False
            return True
        return False

    # ── 路径 ───────────────────────────────────────────────────────────────
    CACHE_DIR            = 'database/system_data/factors_cache'  # 旧版共享缓存；历史模型继续使用
    # 因子公式契约版本。公式语义变化时必须升级，并写入独立缓存目录的 manifest；
    # 禁止原地覆盖旧缓存，否则历史模型的训练/推理输入会静默漂移。
    # 2026-08-14 两处语义变化：
    #   ① 价格复权换成 preclose/close 累乘的**完整**前复权（旧的稀疏
    #      adjust_factor + bfill/ffill 只覆盖 42.5% 除权事件，污染全部滚动窗口因子）；
    #   ② 新增 4 列业绩预告 fc_*（forecast 族）。
    # 2026-08-18 一处语义变化（T115 晋级后把公式搬进生产计算器）：
    #   ③ 新增 5 列指数相对因子 idx_*（index_rel 族），见
    #      core/factors/index_relative_factors.py。必须 bump：不 bump 的话，
    #      改动前后建出来的缓存版本串一模一样却列数不同，正是这套契约要防的
    #      「同名不同公式」静默错配。
    FACTOR_DEFINITION_VERSION = '2026-08-18-fwdadjust-preclose-forecast-idxrel-v1'
    CACHE_MANIFEST_NAME = 'factor_cache_manifest.json'
    SAVE_DIR             = 'models'                              # 模型保存目录

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
    N_FEATURES_TO_SELECT = 80  # 进一步增加特征数，给模型更多信息

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
