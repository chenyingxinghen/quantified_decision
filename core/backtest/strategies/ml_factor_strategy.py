"""
ML因子策略（回测版本）
将ML因子模型集成到新的回测框架
回测时完全依赖训练阶段生成的因子缓存，不再实时计算特征工程
"""
import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import talib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from core.backtest.strategy import BaseStrategy, StrategySignal
from core.factors.ml_factor_model import MLFactorModel
import config.strategy_config as sc
import config.factor_config as fc
from config import DATABASE_PATH, PROJECT_ROOT, SUPPORTED_MARKETS

class MLFactorBacktestStrategy(BaseStrategy):
    """ML因子回测策略
    
    回测时完全依赖训练阶段生成的因子缓存。
    """
    
    def __init__(self,
                 model_path: str,
                 min_confidence: float = sc.ML_FACTOR_MIN_CONFIDENCE,
                 use_cache: bool = True,
                 cache_dir: str = None,
                 norm_stats_path: str = None,
                 risk_min_price: Optional[float] = sc.ML_FACTOR_RISK_MIN_PRICE,
                 risk_exclude_st: bool = sc.ML_FACTOR_RISK_EXCLUDE_ST,
                 max_positions: int = None,
                 max_rel_atr: Optional[float] = None,
                 regime_filter: str = 'off',
                 risk_penalty_lambda: float = 0.0,
                 risk_penalty_feature: str = 'max_drawdown_20',
                 risk_penalty_direction: str = 'low',
                 ensemble_model_paths: Optional[List[str]] = None,
                 preload_start: Optional[str] = None,
                 preload_end: Optional[str] = None,
                 name: str = "ML因子策略"):
        """初始化策略

        风控层参数（T040）：
        - ``max_rel_atr``：入场前波动率上限。候选股 ATR14/close 超过该值直接剔除，
          不占用仓位。None 表示关闭。典型值 0.04~0.06。
        - ``regime_filter``：宏观 regime 空仓开关。
          ``'off'`` 关闭；``'trend'`` 在合成指数处于均线下方且 MA20 斜率为负时
          停止一切新开仓（已有持仓仍由引擎按止损/止盈正常了结）。
          所有判定只用 t 及之前数据，无前视。
        """
        super().__init__(name)
        self.model_path = (
            model_path if os.path.isabs(model_path)
            else os.path.join(PROJECT_ROOT, model_path)
        )
        self.norm_stats_path = (
            norm_stats_path if not norm_stats_path or os.path.isabs(norm_stats_path)
            else os.path.join(PROJECT_ROOT, norm_stats_path)
        )
        self.min_confidence = min_confidence
        self.risk_min_price = risk_min_price
        self.risk_exclude_st = risk_exclude_st
        self.use_cache = use_cache
        # 头部区间大小：回测按模型打分取 Top-K。默认 sc.MAX_POSITIONS；
        # 传入更大值（如 20）即回测"Top-20 头部区间"策略（用户反馈 Top-1 太极端）。
        self.max_positions = max_positions if max_positions is not None else sc.MAX_POSITIONS
        self.max_rel_atr = max_rel_atr
        self.regime_filter = (regime_filter or 'off').lower()
        self.risk_penalty_lambda = float(risk_penalty_lambda or 0.0)
        self.risk_penalty_feature = str(risk_penalty_feature or 'max_drawdown_20')
        self.risk_penalty_direction = str(risk_penalty_direction or 'low').lower()
        self.ensemble_model_paths = [
            p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
            for p in (ensemble_model_paths or []) if p
        ]
        self.ensemble_models = []
        if self.risk_penalty_lambda < 0:
            raise ValueError('risk_penalty_lambda 必须 >= 0')
        if self.risk_penalty_direction not in {'high', 'low'}:
            raise ValueError("risk_penalty_direction 必须为 'high' 或 'low'")
        self._risk_on_dates = None      # np.ndarray[str]，regime 允许开仓的交易日
        self._risk_on_flags = None      # np.ndarray[bool]，与上者同序
        self._regime_blocked_days = 0
        self._volcap_rejected = 0
        
        if cache_dir is None:
            cache_dir = fc.TrainingConfig.CACHE_DIR
        self.cache_dir = cache_dir

        # R2（2026-08-12）：因子面板按回测窗口裁剪日期。
        # 全量常驻 = 5,448 只 × 约 2,525 行 × 219 列 float32 ≈ 12 GB，而回测只用窗口内
        # 的 484 个交易日。裁剪后约 2.3 GB，多种子并行才不会打爆内存（08-08 事故根因）。
        # 数值完全不变：多留窗口起点之前的 1 行，_get_factor_row_array 的
        # searchsorted(..., 'right') - 1 在窗口内取到的行与全量时逐位相同。
        self.preload_start = preload_start
        self.preload_end = preload_end
        
        self.model = None
        self._factors_cache = {}  # 内存缓存，用于存放 parquet 加载全量因子数据
        self._factor_dates_cache = {}
        self._factor_matrix_cache = {}
        self._finance_history = {}
        self._warned_stocks = set()
    
    def initialize(self, **kwargs):
        """初始化策略 - 预定全量缓存以极大提升速度"""
        super().initialize(**kwargs)
        self._active_stock_codes = set(kwargs.get('stock_codes') or [])
        
        # 0. 预加载所有 PIT 筛选所需的基本面指标 (性能优化核心)
        print(f"正在进行 PIT 数据全量预缓存...")
        self._precompute_pit_data()
        
        # 智能加载模型
        from core.factors.ml_factor_model import MLFactorModel, EnsembleFactorModel
        def _load_smart_model(target_path):
            if os.path.isdir(target_path):
                pkls = [
                    os.path.join(target_path, f)
                    for f in os.listdir(target_path)
                    if f.endswith('_factor_model.pkl') or f == 'ensemble_factor_model.pkl'
                ]
                if pkls:
                    latest_pkl = sorted(pkls, key=os.path.getmtime)[-1]
                    return _load_smart_model(latest_pkl)
                return None
            if not os.path.exists(target_path): return None
            # NAMGateModel（NAM 专家 + 宏观 regime 门控）存档需先嗅探再加载，
            # 否则会被 EnsembleFactorModel.load_model 误吞。
            try:
                from core.factors.nam_gate_model import NAMGateModel
                if NAMGateModel.is_nam_gate_archive(target_path):
                    m = NAMGateModel()
                    m.load_model(target_path)
                    print(f"  已识别 NAMGateModel: {target_path}")
                    return m
            except Exception as _e:
                print(f"  NAMGate 嗅探跳过: {_e}")
            try:
                return EnsembleFactorModel.load_model(target_path)
            except:
                try:
                    m = MLFactorModel(); m.load_model(target_path); return m
                except: return None

        self.model = _load_smart_model(self.model_path)
        if self.model is None: raise ValueError(f"无法加载模型: {self.model_path}")
        self.ensemble_models = [self.model]
        for _path in self.ensemble_model_paths:
            _m = _load_smart_model(_path)
            if _m is None:
                raise ValueError(f"无法加载集成子模型: {_path}")
            if list(getattr(_m, 'feature_names', [])) != list(getattr(self.model, 'feature_names', [])):
                raise ValueError(f"集成子模型特征顺序不一致: {_path}")
            self.ensemble_models.append(_m)
        if len(self.ensemble_models) > 1:
            print(f"  已启用横截面分位集成: {len(self.ensemble_models)} 个模型")

        # NAMGateModel 需要逐日的宏观 regime 向量作为门控输入，在此一次性挂载
        self.is_nam_gate = self.model.__class__.__name__ == 'NAMGateModel'
        if self.is_nam_gate:
            try:
                from config import DATABASE_PATH as _DB
                from core.factors.regime_features import build_regime_matrix
                _rm = build_regime_matrix(_DB)
                for _m in self.ensemble_models:
                    if _m.__class__.__name__ == 'NAMGateModel':
                        _m.attach_regime(_rm)
                print(f"  已挂载 regime 矩阵: {_rm.shape[0]} 日 × {_rm.shape[1]} 维")
            except Exception as _e:
                raise RuntimeError(f"NAMGateModel 需要 regime 矩阵，但构建失败: {_e}")

        # 加载归一化统计量（与模型同目录的 norm_stats.pkl）
        import pickle as _pickle
        _model_dir = os.path.dirname(os.path.abspath(self.model_path)) if os.path.isfile(self.model_path) else os.path.abspath(self.model_path)
        _norm_path = (
            os.path.abspath(self.norm_stats_path)
            if self.norm_stats_path
            else os.path.join(_model_dir, 'norm_stats.pkl')
        )
        if os.path.exists(_norm_path):
            try:
                with open(_norm_path, 'rb') as _f:
                    self.norm_stats = _pickle.load(_f)
                print(f"  已加载归一化统计量: {_norm_path}")
            except Exception as _e:
                print(f"  警告: 加载归一化统计量失败: {_e}")
                self.norm_stats = None
        else:
            if self.norm_stats_path:
                raise FileNotFoundError(f"指定的归一化统计量不存在: {_norm_path}")
            # 缺失 norm_stats 会让 skip-rank 连续列以原始量纲进入模型，与训练端
            # 的 robust-sigmoid 归一化严重错配，回测结果完全不可信（曾导致多轮
            # NAM 实验结果雷同且被市值类大数主导）。默认必须硬失败。
            if os.environ.get('ALLOW_MISSING_NORM_STATS') != '1':
                raise FileNotFoundError(
                    f"模型目录缺少 norm_stats.pkl，回测拒绝运行（训练/推理特征尺度会错配）: {_model_dir}\n"
                    f"  → 请用训练脚本重新导出该文件；确需跳过请设 ALLOW_MISSING_NORM_STATS=1")
            print(f"  警告: 模型目录缺少 norm_stats.pkl，跳过全局列归一化: {_model_dir}")
            self.norm_stats = None

        # 风控层：预构建 regime 空仓日历（只用历史，无前视）
        if self.regime_filter != 'off':
            self._build_risk_on_calendar()

        if self.use_cache:
            self._preload_factor_cache()
        
        print(f"策略初始化完成: {self.name} (已启用 PIT 预缓存)")
    
    @staticmethod
    def _risk_adjusted_scores(probs: np.ndarray,
                              risk_values: np.ndarray,
                              penalty_lambda: float,
                              direction: str = 'high') -> np.ndarray:
        """在横截面分位空间施加风险惩罚，并拒绝退化风险列。"""
        from scipy.stats import rankdata as _rankdata

        pred = np.asarray(probs, dtype=float)
        risk = np.asarray(risk_values, dtype=float)
        if pred.ndim != 1 or risk.ndim != 1 or len(pred) != len(risk):
            raise ValueError('模型分数与风险值必须是一维等长数组')
        if not np.all(np.isfinite(risk)):
            raise ValueError('风险惩罚横截面包含非有限值')
        unique_ratio = len(np.unique(risk)) / max(len(risk), 1)
        if len(risk) > 1 and (np.ptp(risk) <= 1e-8 or unique_ratio < 0.05):
            raise RuntimeError(
                f'风险惩罚特征横截面已退化（唯一值占比 {unique_ratio:.2%}），拒绝静默运行'
            )
        if direction not in {'high', 'low'}:
            raise ValueError("direction 必须为 'high' 或 'low'")

        model_pct = _rankdata(pred, method='average') / (len(pred) + 1)
        risk_pct = _rankdata(risk, method='average') / (len(risk) + 1)
        if direction == 'low':
            risk_pct = 1.0 - risk_pct
        return model_pct - float(penalty_lambda) * risk_pct

    def generate_signals(self,
                        current_date: str,
                        market_data: Any,
                        portfolio_state: Dict[str, Any],
                        criteria: Optional[Dict] = None,
                        min_confidence: Optional[float] = None) -> List[StrategySignal]:
        """生成交易信号 (极速版)"""
        signals = []
        existing_positions = portfolio_state.get('positions', {})
        
        # 确定置信度阈值：优先使用参数，其次使用实例属性
        effective_min_confidence = min_confidence if min_confidence is not None else self.min_confidence
        
        # 优先使用 portfolio_state 中的 available_slots (用于实盘选股指定数量)
        # 如果未指定，则根据最大仓位限制计算剩余空位
        if 'available_slots' in portfolio_state:
            available_slots = portfolio_state['available_slots']
        else:
            available_slots = self.max_positions - len(existing_positions)
            
        if available_slots <= 0: return signals

        # 风控层 1：regime 空仓开关。下行 regime 直接放弃当日全部新开仓，
        # 已有持仓仍由引擎按止损/止盈正常了结。
        if self.regime_filter != 'off' and not self._is_risk_on(current_date):
            self._regime_blocked_days += 1
            return signals

        # 1. 获取所有股票列表。风险资格与已有持仓必须在模型打分后处理，
        # 否则会改变横截面排名，导致其他股票的模型输入随持仓/过滤参数漂移。
        all_codes = list(market_data.keys())

        # 筛选逻辑：
        # 1. 回测场景：criteria 为 None，完全使用 sc 配置
        # 2. 后端场景：criteria 由前端传入，前端必须明确传入所有参数（不传或空值则不限制该条件）
        # 3. 自动化场景：criteria 由自动化传入，自动化配置优先，未设置的参数使用 sc 兜底
        if criteria is not None:
            # 后端/自动化场景：criteria 已传入
            filter_criteria = criteria
            should_apply_filter = criteria.get('apply_filter', False)  # 默认 False，必须显式启用
        else:
            # 回测场景：使用 sc 配置
            filter_criteria = {
                'min_market_cap': sc.MIN_MARKET_CAP, 'max_pe': sc.MAX_PE, 'max_zcfzl': sc.MAX_ZCFZL,
                'min_price': sc.MIN_PRICE, 'max_price': sc.MAX_PRICE, 'include_st': sc.INCLUDE_ST,
                'markets': sc.SELECTOR_MARKETS
            }
            should_apply_filter = sc.ENABLE_FUNDAMENTAL_FILTER

        if should_apply_filter:
            # 2. 预筛选 (利用内存快照，无 SQL)。仅在启用过滤时构建，避免无谓遍历全市场。
            info_map = self._get_optimized_info_map(current_date, market_data)
            predict_codes, _ = self._pre_filter_stocks(all_codes, info_map, 
                                                 apply_filter=True, 
                                                 criteria=filter_criteria)
        else:
            predict_codes = all_codes
        
        if not predict_codes: return signals

        # 3. 批量获取当前因子的最新行
        raw_rows = []
        stock_codes_with_data = []
        feature_names = self._get_model_feature_names()
        if not feature_names:
            return signals
        for code in predict_codes:
            feature_row = self._get_factor_row_array(code, current_date)
            if feature_row is None:
                continue
            raw_rows.append(feature_row)
            stock_codes_with_data.append(code)
        
        if not raw_rows: return signals

        # 4. 批量预测。优先保持 numpy 矩阵，避免每日反复构造 DataFrame。
        X_arr = np.vstack(raw_rows).astype(np.float32, copy=False)

        # 横截面归一化 (与训练时精确匹配逻辑保持一致)
        rank_cols_idx = [i for i, col in enumerate(feature_names) if not fc.TrainingConfig.should_skip_rank(col)]

        if rank_cols_idx and len(X_arr) > 1:
            from scipy.stats import rankdata as _rankdata
            ranked = _rankdata(X_arr[:, rank_cols_idx], method='average', axis=0) / (len(X_arr) + 1)
            X_arr[:, rank_cols_idx] = ranked.astype(np.float32)

        # skip_cols：复用训练集 robust 统计量（仅连续型，二值列保留原值）
        norm_stats = getattr(self, 'norm_stats', None)
        if norm_stats is not None:
            import numpy as _np
            skip_col_stats = norm_stats.get('skip_col_stats')
            if skip_col_stats is not None and skip_col_stats.get('robust_global_idx', _np.array([])).size > 0:
                train_factor_names = norm_stats.get('factor_names', [])
                robust_global_idx  = skip_col_stats['robust_global_idx']
                robust_col_names   = [train_factor_names[i] for i in robust_global_idx
                                      if i < len(train_factor_names)]
                feature_pos = {name: idx for idx, name in enumerate(feature_names)}
                present_robust = [c for c in robust_col_names if c in feature_pos]
                if present_robust:
                    col_to_idx = {train_factor_names[i]: j
                                  for j, i in enumerate(robust_global_idx)
                                  if i < len(train_factor_names)}
                    median    = skip_col_stats['median']
                    iqr       = skip_col_stats['iqr']
                    valid_iqr = skip_col_stats['valid_iqr']
                    for col in present_robust:
                        j = col_to_idx[col]
                        arr_idx = feature_pos[col]
                        if valid_iqr[j]:
                            z = (X_arr[:, arr_idx].astype(float) - median[j]) / iqr[j]
                            X_arr[:, arr_idx] = (1.0 / (1.0 + _np.exp(-_np.clip(z, -10, 10)))).astype(_np.float32)
                        else:
                            # 必须与训练端保持一致；训练端零 IQR 列固定为 0.0。
                            X_arr[:, arr_idx] = 0.0
        X_arr = np.nan_to_num(X_arr, nan=0.5, posinf=1.0, neginf=0.0)
        if len(self.ensemble_models) > 1:
            from scipy.stats import rankdata as _rankdata
            _model_ranks = []
            _frame = pd.DataFrame(X_arr, columns=feature_names)
            for _m in self.ensemble_models:
                if _m.__class__.__name__ == 'NAMGateModel':
                    _m.set_context_date(current_date)
                    _pred = np.asarray(_m.predict(_frame), dtype=float)
                elif getattr(_m, 'models', None):
                    _pred = np.asarray(_m.predict(_frame), dtype=float)
                else:
                    _pred = np.asarray(_m.predict(X_arr), dtype=float)
                # 每个模型独立分位化，消除初始化引起的输出温度/尺度差异。
                _model_ranks.append(_rankdata(_pred, method='average') / (len(_pred) + 1))
            probs = np.mean(np.vstack(_model_ranks), axis=0)
        elif getattr(self, 'is_nam_gate', False):
            # 门控依赖"当前交易日"的市场状态，必须在打分前显式告知
            self.model.set_context_date(current_date)
            probs = self.model.predict(pd.DataFrame(X_arr, columns=feature_names))
        elif getattr(self.model, 'models', None):
            probs = self.model.predict(pd.DataFrame(X_arr, columns=feature_names))
        else:
            probs = self.model.predict(X_arr)
        
        # 5. 生成信号。可选的下行风险惩罚采用横截面分位组合：
        # adjusted_score = model_percentile - lambda * risk_percentile。
        # 不直接从 NAM 原始分数减风险值，因为不同随机种子的 NAM 输出温度/量纲并不一致；
        # 分位化后两者都落在 (0,1)，lambda 才具有跨模型可比含义。
        # risk 值来自当日 PIT 因子缓存，只使用 t 及之前数据，无前视。
        adjusted_scores = np.asarray(probs, dtype=float).copy()
        risk_values = None
        if self.risk_penalty_lambda > 0:
            try:
                risk_idx = feature_names.index(self.risk_penalty_feature)
            except ValueError as exc:
                raise ValueError(
                    f"风险惩罚特征不在模型输入中: {self.risk_penalty_feature}"
                ) from exc
            # X_arr 已完成与训练一致的横截面 rank；风险方向由参数显式声明。
            risk_values = X_arr[:, risk_idx].astype(float)
            adjusted_scores = self._risk_adjusted_scores(
                adjusted_scores,
                risk_values,
                self.risk_penalty_lambda,
                self.risk_penalty_direction,
            )

        candidates = []
        for i, code in enumerate(stock_codes_with_data):
            confidence = float(probs[i] * 100)
            # min_confidence <= 0 语义为"不设阈值"。排序类模型（NAM/LambdaRank）的输出
            # 无界且可为负，负分只代表横截面靠后而非无效，若沿用概率语义做 `< 0` 截断，
            # 会把整个候选池砍空（曾导致某次回测仅成交 1 笔）。
            if effective_min_confidence > 0 and confidence < effective_min_confidence: continue
            if code in existing_positions: continue

            # 仅在完整横截面完成归一化和预测后执行 PIT 风险资格判断。
            if self.risk_min_price is not None or self.risk_exclude_st:
                if not self._passes_basic_risk_filter(
                    market_data.get_bar(code),
                    min_price=self.risk_min_price,
                    exclude_st=self.risk_exclude_st,
                ):
                    continue
            
            # 使用 md5 哈希在概率相同时保持排序稳定
            tie_breaker = int(hashlib.md5(code.encode()).hexdigest(), 16) % 1000 / 100000.0
            # 保持 StrategySignal.confidence 的百分制历史语义；lambda 本身仍作用在 0~1 分位空间。
            score = (float(adjusted_scores[i]) * 100.0
                     if self.risk_penalty_lambda > 0 else confidence)
            candidates.append({
                'code': code,
                'score': score + tie_breaker,
                'prob': probs[i],
                'risk_value': None if risk_values is None else float(risk_values[i]),
            })
            
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 诊断插桩：设置 MLFS_DEBUG_RANK=<文件路径> 时逐日落盘候选池规模与 Top10 打分，
        # 用于验证"不同模型是否真的产生不同选股"。默认关闭，零开销。
        _dbg = os.environ.get('MLFS_DEBUG_RANK')
        if _dbg:
            with open(_dbg, 'a', encoding='utf-8') as _fh:
                _top = [(c['code'], round(float(c['prob']), 6)) for c in candidates[:10]]
                _fh.write(f"{current_date}\tn_pred={len(stock_codes_with_data)}"
                          f"\tn_cand={len(candidates)}\tslots={available_slots}\ttop10={_top}\n")

        # 风控层 2：入场前波动率上限。按打分顺序逐个体检，被剔除的高波动股
        # 不占用仓位（由下一名顺延），因此需要遍历全部候选而非前 N 名。
        for cand in candidates:
            if len(signals) >= available_slots:
                break
            code = cand['code']
            bar = market_data.get_bar(code)
            if bar is None: continue
            
            # 获取 ATR 需要历史数据
            hist_df = market_data[code]
            atr = self._calculate_atr(hist_df)
            if self.max_rel_atr is not None:
                close_px = float(bar['close']) if np.isfinite(bar['close']) else float('nan')
                rel_atr = (atr / close_px) if (atr > 0 and np.isfinite(close_px) and close_px > 0) else float('nan')
                # 无法计算波动率的标的一律拒绝：宁可少开仓，不可裸奔
                if not np.isfinite(rel_atr) or rel_atr > self.max_rel_atr:
                    self._volcap_rejected += 1
                    continue
            signals.append(StrategySignal(
                stock_code=code, signal_type='buy', timestamp=current_date, 
                price=bar['close'], confidence=cand['score'], 
                stop_loss=bar['close'] - sc.ATR_STOP_MULTIPLIER * atr, 
                take_profit=bar['close'] + sc.ATR_TARGET_MULTIPLIER * atr,
                metadata={
                    'strategy': 'ml_factor_integrated',
                    'prediction': cand['prob'],
                    'adjusted_score': cand['score'],
                    'risk_penalty_feature': self.risk_penalty_feature,
                    'risk_value': cand.get('risk_value'),
                }
            ))
        return signals

    def _build_risk_on_calendar(self):
        """构建逐日 risk-on 标记：False 的交易日禁止一切新开仓。

        判据（``regime_filter='trend'``）：合成全市场指数同时满足
        ``trend_ma20_dev >= 0``（指数在 20 日均线上方）与
        ``trend_ma20_slope >= 0``（均线本身在上行）才视为 risk-on。
        两列均由 ``build_regime_matrix(..., return_raw=True)`` 提供，
        全部为 t 及之前的滚动量，不含未来信息。
        """
        from core.factors.regime_features import build_regime_matrix
        raw = build_regime_matrix(DATABASE_PATH, normalize=False, return_raw=True)
        if raw is None or raw.empty:
            raise RuntimeError('regime_filter 已启用，但市场状态矩阵为空')

        if self.regime_filter == 'trend':
            dev = raw.get('trend_ma20_dev')
            slope = raw.get('trend_ma20_slope')
            if dev is None or slope is None:
                raise RuntimeError('regime 矩阵缺少 trend_ma20_dev / trend_ma20_slope')
            flags = (dev.to_numpy() >= 0.0) & (slope.to_numpy() >= 0.0)
        else:
            raise ValueError(f'未知 regime_filter: {self.regime_filter}')

        self._risk_on_dates = raw.index.strftime('%Y-%m-%d').to_numpy()
        self._risk_on_flags = np.asarray(flags, dtype=bool)
        on_ratio = float(self._risk_on_flags.mean())
        print(f"  regime 空仓开关已启用({self.regime_filter}): "
              f"历史 risk-on 日占比 {on_ratio:.1%} / {len(self._risk_on_flags)} 日")

    def _is_risk_on(self, current_date: str) -> bool:
        """当前交易日是否允许开仓。日期缺失时回溯最近一个有效判定。"""
        if self._risk_on_flags is None:
            return True
        idx = np.searchsorted(self._risk_on_dates, current_date, side='right') - 1
        if idx < 0:
            return False  # 热身期信息不足，保守空仓
        return bool(self._risk_on_flags[idx])

    def _relative_atr(self, hist_df: pd.DataFrame, close: float) -> float:
        """相对波动率 = ATR14 / 收盘价。无法计算时返回 nan。"""
        if hist_df is None or close is None or not np.isfinite(close) or close <= 0:
            return float('nan')
        atr = self._calculate_atr(hist_df)
        if atr <= 0:
            return float('nan')
        return float(atr / close)

    def _precompute_pit_data(self):
        """预加载元数据，消除循环内的 SQL 压力"""
        db_dir = os.path.dirname(DATABASE_PATH)
        conn = sqlite3.connect(DATABASE_PATH)
        for db in ['stock_meta.db', 'stock_finance.db']:
            path = os.path.join(db_dir, db)
            if os.path.exists(path): conn.execute(f"ATTACH DATABASE '{path}' AS {db.split('_')[1].split('.')[0]}")
        try:
            self._meta_map = pd.read_sql_query("SELECT code, code_name AS name FROM meta.stock_basic", conn).set_index('code')['name'].to_dict()
            self._all_finance_df = pd.read_sql_query("""
                SELECT p.code, p.pub_date, p.stat_date, p.epsTTM AS EPSJB, p.totalShare, b.liabilityToAsset AS ZCFZL
                FROM finance.profit_ability p
                LEFT JOIN finance.balance_ability b ON p.code = b.code AND p.stat_date = b.stat_date
                WHERE p.pub_date IS NOT NULL AND p.pub_date != ''
            """, conn)
            num_cols = ['EPSJB', 'totalShare', 'ZCFZL']
            for col in num_cols:
                self._all_finance_df[col] = pd.to_numeric(self._all_finance_df[col], errors='coerce').astype('float32')
            self._all_finance_df = self._all_finance_df.sort_values('pub_date')
            self._finance_history = {}
            for code, group in self._all_finance_df.groupby('code', sort=False):
                dates = group['pub_date'].astype(str).str[:10].to_numpy()
                values = group[num_cols].to_numpy(dtype=np.float32, copy=True)
                self._finance_history[code] = (dates, values)
        finally: conn.close()

    def _get_optimized_info_map(self, target_date: str, market_data: Any) -> Dict[str, Dict]:
        """极速信息映射逻辑，复用预缓存信息"""
        info_map = {}
        
        for code in market_data.keys():
            bar = market_data.get_bar(code)
            if bar is None: continue
            fin_hist = getattr(self, '_finance_history', {}).get(code)
            eps = ts = zcfzl = None
            if fin_hist is not None:
                fin_dates, fin_values = fin_hist
                fin_idx = np.searchsorted(fin_dates, target_date, side='right') - 1
                if fin_idx >= 0:
                    eps = fin_values[fin_idx, 0]
                    ts = fin_values[fin_idx, 1]
                    zcfzl = fin_values[fin_idx, 2]
            # 组装基本面字典供筛选
            raw_close = float(bar.get('raw_close', bar['close']))
            info_map[code] = {
                'name': self._meta_map.get(code, '-'), 'is_st': int(bar.get('is_st', 0)),
                'pe_ratio': (raw_close/eps if eps and eps>0 else None),
                'zcfzl': zcfzl, 'current_price': raw_close,
                'market_cap': (raw_close*ts/1e8 if ts else None)
            }
        return info_map

    def _preload_factor_cache(self):
        """预加载回测所需因子列，并建立按日期二分查找的缓存。"""
        feature_names = self._get_model_feature_names()
        if not feature_names:
            return
        if not os.path.isdir(self.cache_dir):
            return

        active_codes = getattr(self, '_active_stock_codes', set())
        if active_codes:
            files = [
                f'{code}_factors.parquet'
                for code in active_codes
                if os.path.exists(os.path.join(self.cache_dir, f'{code}_factors.parquet'))
            ]
        else:
            files = [f for f in os.listdir(self.cache_dir) if f.endswith('_factors.parquet')]
        if not files:
            return

        scope = "当前回测股票池" if active_codes else "全部缓存"
        lo_key = str(self.preload_start)[:10] if self.preload_start else None
        hi_key = str(self.preload_end)[:10] if self.preload_end else None
        window = f", 日期裁剪 {lo_key or '-inf'}~{hi_key or '+inf'}" if (lo_key or hi_key) else ""
        print(f"正在预加载因子缓存({scope}{window}): {len(files)} 个 parquet...")

        def _load_one(filename):
            code = filename[:-len('_factors.parquet')]
            path = os.path.join(self.cache_dir, filename)
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(path)
                schema_names = set(pf.schema.names)
                if 'date' not in schema_names:
                    return None
                read_cols = ['date'] + [c for c in feature_names if c in schema_names]
                df = pd.read_parquet(path, columns=read_cols)
                if df.empty or 'date' not in df.columns:
                    return None

                dates = df['date'].astype(str).str[:10].to_numpy()
                order = np.argsort(dates, kind='mergesort')
                if not np.all(order == np.arange(len(order))):
                    df = df.iloc[order].reset_index(drop=True)
                    dates = dates[order]

                # R2：按回测窗口裁剪。lo 多留 1 行（窗口起点之前最近的一行），
                # 以保证 PIT 取行 searchsorted(dates, d, 'right') - 1 与全量一致。
                if lo_key is not None or hi_key is not None:
                    lo = 0
                    if lo_key is not None:
                        lo = max(0, int(np.searchsorted(dates, lo_key, side='right')) - 1)
                    hi = len(dates)
                    if hi_key is not None:
                        hi = int(np.searchsorted(dates, hi_key, side='right'))
                    if hi <= lo:
                        return None
                    if lo > 0 or hi < len(dates):
                        df = df.iloc[lo:hi]
                        dates = dates[lo:hi]

                matrix = np.full((len(df), len(feature_names)), 0.5, dtype=np.float32)
                for col_idx, col in enumerate(feature_names):
                    if col in df.columns:
                        matrix[:, col_idx] = pd.to_numeric(df[col], errors='coerce').fillna(0.5).to_numpy(dtype=np.float32)
                return code, dates, matrix
            except Exception:
                return None

        from concurrent.futures import ThreadPoolExecutor
        workers = min(8, max(1, len(files)))
        loaded = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for item in executor.map(_load_one, files):
                if item is None:
                    continue
                code, dates, matrix = item
                self._factor_dates_cache[code] = dates
                self._factor_matrix_cache[code] = matrix
                loaded += 1
        print(f"  因子缓存预加载完成: {loaded}/{len(files)}")

    def _get_factor_row_array(self, stock_code: str, current_date: str) -> Optional[np.ndarray]:
        """按股票和日期快速取得 PIT 特征行。"""
        dates = self._factor_dates_cache.get(stock_code)
        matrix = self._factor_matrix_cache.get(stock_code)
        if dates is not None and matrix is not None:
            idx = np.searchsorted(dates, current_date, side='right') - 1
            if idx >= 0:
                return matrix[idx]

        factors = self._get_factors(stock_code, None, current_date)
        if factors is None or factors.empty:
            return None
        row = factors.drop(columns=['date'], errors='ignore').iloc[-1]
        return row.reindex(self._get_model_feature_names()).fillna(0.5).to_numpy(dtype=np.float32)

    def _get_model_feature_names(self) -> List[str]:
        """获取模型需要的特征列，兼容单模型与集成模型。"""
        names = getattr(self.model, 'feature_names', None)
        if names:
            return list(names)
        models = getattr(self.model, 'models', None)
        if models:
            ordered = []
            seen = set()
            for model in models:
                for name in getattr(model, 'feature_names', []) or []:
                    if name not in seen:
                        ordered.append(name)
                        seen.add(name)
            return ordered
        return []

    def _pre_filter_stocks(self, all_codes: List[str], info_map: Dict[str, Dict], apply_filter: bool, criteria: Dict) -> Tuple[List[str], Dict]:
        """
        股票池预筛选逻辑
        
        参数优先级说明：
        - 后端场景：criteria 中的参数由前端传入，None 表示不限制该条件
        - 回测场景：criteria 使用 sc 配置的默认值
        - 自动化场景：criteria 使用 automation_config 优先，未设置则使用 sc 兜底
        """
        if not apply_filter: return all_codes, {}
        passed = []
        
        def _get_val(k, default):
            v = criteria.get(k)
            return v if v is not None else default

        # 预计算允许的市场前缀
        markets_filter = _get_val('markets', [])
        allowed_prefixes = []
        if markets_filter:
            for m in markets_filter:
                p = SUPPORTED_MARKETS.get(m, {}).get('prefixes')
                if p: 
                    allowed_prefixes.extend(p)
                else:
                    # 兼容性处理：支持交易所级别代码 (如 'sh' -> ['60', '68'])
                    for m_info in SUPPORTED_MARKETS.values():
                        if m_info.get('code') == m:
                            allowed_prefixes.extend(m_info.get('prefixes', []))
        allowed_prefixes = tuple(allowed_prefixes) if allowed_prefixes else None
            
        for code in all_codes:
            info = info_map.get(code)
            if not info: continue
            
            # 1. 市场类型筛选
            if allowed_prefixes and not str(code).startswith(allowed_prefixes):
                continue

            # 2. ST股及退市搜索
            # 股票名称来自当前元数据，不具备 PIT 语义，回测中不能用“退”字样
            # 反向污染历史。风险过滤只依赖当日行情中的 is_st。
            if not _get_val('include_st', sc.INCLUDE_ST) and info['is_st'] == 1: continue
            
            # 3. PE 筛选
            pe = info.get('pe_ratio')
            max_pe = _get_val('max_pe', sc.MAX_PE if sc.MAX_PE is not None else float('inf'))
            if pe is not None and (pe <= 0 or pe > max_pe): continue
            
            # 4. 价格筛选
            price = info['current_price']
            if price < _get_val('min_price', sc.MIN_PRICE if sc.MIN_PRICE is not None else 0) or \
               price > _get_val('max_price', sc.MAX_PRICE if sc.MAX_PRICE is not None else float('inf')): continue
            
            # 5. 市值筛选
            mkt_cap = info.get('market_cap')
            if mkt_cap is not None:
                if mkt_cap < _get_val('min_market_cap', 0): continue
            
            # 6. 资产负债率筛选
            zcfzl = info.get('zcfzl')
            if zcfzl is not None:
                if zcfzl > _get_val('max_zcfzl', float('inf')): continue
                
            passed.append(code)
        return passed, {}

    @staticmethod
    def _passes_basic_risk_filter(
        bar: Optional[Dict[str, Any]],
        min_price: Optional[float],
        exclude_st: bool,
    ) -> bool:
        """使用当日行情执行最低价和 ST 风险过滤。"""
        if bar is None:
            return False
        if exclude_st and int(bar.get('is_st', 0)) == 1:
            return False
        if min_price is not None:
            price = float(bar.get('raw_close', bar.get('close', np.nan)))
            if not np.isfinite(price) or price < min_price:
                return False
        return True

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """ATR 计算"""
        if len(data) < period + 1: return 0.0
        try:
            atr_series = talib.ATR(data['high'].values.astype(np.float64), 
                                   data['low'].values.astype(np.float64), 
                                   data['close'].values.astype(np.float64), 
                                   timeperiod=period)
            val = atr_series[-1]
            return float(val) if np.isfinite(val) else 0.0
        except: return 0.0

    def _get_factors(self, stock_code: str, stock_data: pd.DataFrame, current_date: str) -> Optional[pd.DataFrame]:
        """获取 PIT 因子行"""
        cached_factors = self._load_factors_from_cache(stock_code)
        if cached_factors is None or 'date' not in cached_factors.columns: return None
        target_dt = datetime.strptime(current_date, '%Y-%m-%d')
        # 获取不晚于当前日期的最新因子
        factors = cached_factors[pd.to_datetime(cached_factors['date']) <= target_dt]
        if factors.empty: return None
        factors = factors.iloc[[-1]]
        
        feature_names = self._get_model_feature_names()
        if self.model and feature_names:
            # 填充缺失列为0，保持特征列顺序与模型一致
            missing = [f for f in feature_names if f not in factors.columns]
            if missing:
                missing_df = pd.DataFrame(0.5, index=factors.index, columns=missing)
                factors = pd.concat([factors, missing_df], axis=1)
            f_cols = feature_names + (['date'] if 'date' not in feature_names else [])
            factors = factors[[c for c in f_cols if c in factors.columns]]
        return factors

    def _load_factors_from_cache(self, stock_code: str) -> Optional[pd.DataFrame]:
        """从磁盘加载 parquet 文件并维护内存缓存"""
        if stock_code in self._factors_cache: return self._factors_cache[stock_code]
        cache_file = os.path.join(self.cache_dir, f'{stock_code}_factors.parquet')
        if not os.path.exists(cache_file): return None
        try:
            factors = pd.read_parquet(cache_file)
            # 容量上限提升至 10,000，足以容纳全市场股票。只有在内存极度受限且股票池巨大时才会触发清理。
            if len(self._factors_cache) >= 10000:
                # 简单 FIFO 淘汰
                first_key = next(iter(self._factors_cache))
                self._factors_cache.pop(first_key)
            self._factors_cache[stock_code] = factors
            return factors
        except: return None

    def select_for_live(self,
                        db_path: str,
                        top_n: int = 10,
                        lookback_days: int = 500,
                        criteria: Optional[Dict] = None) -> List[Dict]:
        """
        实盘选股入口，完全复用 generate_signals 逻辑，保证与回测一致。

        返回列表，每项包含：
            stock_code, confidence, current_price, stop_loss, take_profit
        """
        import sqlite3
        from datetime import datetime, timedelta

        today = datetime.now().strftime('%Y-%m-%d')

        # --- 构造轻量 LiveMarketData 适配器 ---
        class LiveMarketData:
            """从数据库读取最新行情，提供与回测 MarketSnapshot 相同的接口"""
            def __init__(self, db_path: str, lookback_days: int):
                self._db_path = db_path
                self._lookback_days = lookback_days
                self._cache: Dict[str, pd.DataFrame] = {}
                self._bar_cache: Dict[str, Optional[Dict]] = {}
                self._codes: Optional[List[str]] = None

            def _load(self, code: str) -> Optional[pd.DataFrame]:
                if code in self._cache:
                    return self._cache[code]
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=self._lookback_days)).strftime('%Y-%m-%d')
                try:
                    conn = sqlite3.connect(self._db_path)
                    df = pd.read_sql_query(
                        """SELECT k.date, k.open, k.high, k.low, k.close, k.volume,
                                  k.amount, k.turnover_rate, k.is_st, a.fore_adjust_factor
                           FROM daily_data k
                           LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
                           WHERE k.code = ? AND k.date >= ? AND k.date <= ?
                           ORDER BY k.date ASC""",
                        conn, params=(code, start_date, end_date)
                    )
                    conn.close()
                    if df.empty or len(df) < 35:
                        return None
                    self._cache[code] = df
                    return df
                except Exception:
                    return None

            def get_bar(self, code: str) -> Optional[Dict]:
                if code in self._bar_cache:
                    return self._bar_cache[code]
                df = self._load(code)
                if df is None:
                    self._bar_cache[code] = None
                    return None
                row = df.iloc[-1]
                bar = {c: row[c] for c in df.columns}
                # is_st 已从数据库读取，无需覆盖
                self._bar_cache[code] = bar
                return bar

            def __getitem__(self, code: str) -> Optional[pd.DataFrame]:
                return self._load(code)

            def keys(self) -> List[str]:
                if self._codes is not None:
                    return self._codes
                try:
                    conn = sqlite3.connect(self._db_path)
                    rows = pd.read_sql_query(
                        "SELECT DISTINCT code FROM daily_data WHERE date >= date('now', '-30 days')",
                        conn
                    )
                    conn.close()
                    self._codes = rows['code'].tolist()
                except Exception:
                    self._codes = []
                return self._codes

        market_data = LiveMarketData(db_path, lookback_days)

        # 复用 generate_signals，portfolio_state 传空持仓、足够的 slots
        portfolio_state = {'positions': {}, 'available_slots': top_n}
        
        # 调用信号生成逻辑，显式传递 criteria
        # 内部会优先使用传入的 criteria 进行筛选，且不使用 sc 中的默认覆盖值
        signals = self.generate_signals(today, market_data, portfolio_state, 
                                        criteria=criteria)

        # 转换为与 select_stocks 兼容的字典格式
        results = []
        # signals 已经是经过排序和 top_n 截取的了 (通过 available_slots)
        for sig in signals:
            results.append({
                'stock_code': sig.stock_code,
                'confidence': sig.confidence,
                'current_price': sig.price,
                'stop_loss': sig.stop_loss,
                'take_profit': sig.take_profit,
            })
        return results

    def cleanup(self):
        """清理缓存"""
        self._factors_cache.clear()
        if self.regime_filter != 'off' or self.max_rel_atr is not None:
            print(f"  风控层统计: regime 空仓 {self._regime_blocked_days} 日, "
                  f"波动率上限剔除 {self._volcap_rejected} 次候选")
        print(f"策略清理完成: {self.name}")
