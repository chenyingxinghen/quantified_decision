"""
analysis/common.py — 共享工具

提供：
  - 模型与数据集加载（含 parquet 缓存 / 合成数据回退，便于无 DB 环境冒烟测试）
  - 横截面归一化（与训练完全一致）
  - 模型打分预测
  - 特征归类（财务 / 交易 / 估值 / 其他）
  - 牛 / 熊 / 震荡 市场状态划分（以等权截面收益为市场代理）

所有对外函数都尽量无副作用，便于各分析模块独立复用。
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 项目内依赖（云端训练镜像中均存在）────────────────────────────────────
from core.factors.ml_factor_model import MultiObjectiveFactorModel  # noqa: E402
from config.jydb_config import DATABASE_PATH  # noqa: E402
from config.factor_config import TrainingConfig  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
# 1. 模型与数据集加载
# ════════════════════════════════════════════════════════════════════════
def load_model(model_path: str):
    """加载多目标因子模型。

    同时支持 GBM 树模型（MultiObjectiveFactorModel）与神经网络模型
    （MultiObjectiveNeuralModel）。优先按 GBM 加载，若 format 不匹配
    （神经网络保存的 multi_objective_neural_model）则安全降级到神经网络加载。
    """
    try:
        return MultiObjectiveFactorModel.load_model(model_path)
    except Exception:
        # 神经网络模型（无 torch 时会在此抛 ImportError，属预期，调用方需保证环境）
        from core.neural.nn_models import MultiObjectiveNeuralModel
        return MultiObjectiveNeuralModel.load_model(model_path)


def load_full_dataset_from_parquet(path: str):
    """从 parquet 缓存恢复 full_dataset。

    返回: (X, y, returns, factor_names, dates, unbuyable_mask,
           limit_groups, path_scores, is_st_arr, w_sig_arr, codes)
    """
    store = pd.read_parquet(path)
    meta = pickle.loads(store.attrs["meta"])
    shapes = meta.get("shapes")
    arrays = []
    for i in range(meta["n_arrays"]):
        arr = store[f"arr_{i}"].to_numpy()
        # 二维数组（特征矩阵 X）落盘时被展平，读取时按记录的形状还原
        if shapes is not None and len(shapes[i]) == 2:
            arr = arr.reshape(shapes[i])
        arrays.append(arr)
    factor_names = meta["factor_names"]
    # codes 是 object 列，需从单独列取回
    codes = store["codes"].to_numpy()
    # 把 codes 放回第 10 位（与 prepare_dataset 顺序一致）
    out = list(arrays)
    out.insert(10, codes)
    # 加载后统一特征矩阵为 float32（与 build/评分一致，降低内存占用）
    out[0] = np.ascontiguousarray(out[0], dtype=np.float32)
    return tuple(out)


def save_full_dataset_to_parquet(path: str, full_dataset) -> None:
    """把 full_dataset 存成单文件 parquet（便于分析阶段复用，避免重复拉库）。"""
    (X, y, returns, factor_names, dates, unbuyable_mask,
     limit_groups, path_scores, is_st_arr, w_sig_arr, codes) = full_dataset
    arrays = [X, y, returns, np.asarray(factor_names, dtype=object),
              np.asarray(dates, dtype=object), unbuyable_mask,
              limit_groups, path_scores, is_st_arr, w_sig_arr]
    cols = {}
    shapes = []
    for i, a in enumerate(arrays):
        a = np.asarray(a)
        shapes.append(a.shape)
        if a.ndim == 1:
            cols[f"arr_{i}"] = pd.Series(a)
        elif a.ndim == 2:
            # X 是二维特征矩阵，DataFrame 单列必须是一维；展平存储并记录形状以便恢复
            cols[f"arr_{i}"] = pd.Series(a.reshape(-1))
        else:
            raise ValueError(f"不支持的数组维度 ndim={a.ndim} (arr_{i})")
    cols["codes"] = pd.Series(codes)
    df = pd.DataFrame(cols)
    meta = {
        "n_arrays": len(arrays),
        "factor_names": list(factor_names),
        "shapes": shapes,
    }
    df.attrs["meta"] = pickle.dumps(meta)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


def build_full_dataset(
    trainer,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stocks: int = 200,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
):
    """复用训练管线的数据集构建逻辑（与 scripts/train_model.py 一致），并缓存到 parquet。

    参数:
        trainer:        MLModelTrainer 实例
        start_date/end_date: 训练标签区间（默认取训练配置）
        stocks:         股票数量（云端用全量；本地冒烟测试可减小以避免 OOM）
        cache_path:     缓存 parquet 路径；命中且非强制重建时直接返回
        force_rebuild:  True 时忽略缓存重建
    """
    if cache_path and os.path.exists(cache_path) and not force_rebuild:
        print(f"[dataset] 命中缓存: {cache_path}")
        return load_full_dataset_from_parquet(cache_path)

    from datetime import datetime, timedelta
    from core.data.jydb_market_etl import JYDBMarketETL

    train_end_date = end_date or (
        datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)
    ).strftime("%Y-%m-%d")
    train_start_date = start_date or (
        datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_TRAINING)
    ).strftime("%Y-%m-%d")

    stock_codes_all = JYDBMarketETL.get_stock_list(
        DATABASE_PATH, as_of_date=train_end_date, limit=stocks
    )
    trainer_stocks = stock_codes_all[:stocks]
    print(f"[dataset] 股票池: {len(trainer_stocks)} 只, 区间 {train_start_date}~{train_end_date}")

    stocks_data = trainer.load_label_data(trainer_stocks, train_start_date, train_end_date)
    full_dataset = trainer.prepare_dataset(
        stocks_data,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        include_fundamentals=True,
        n_jobs=getattr(TrainingConfig, "N_JOBS_FACTOR_CALC", 8),
        use_factor_cache_only=True,
        return_codes=True,
    )
    # 特征矩阵 X 降为 float32：与训练归一化 (Xn 为 float32) 一致，且把后续
    # save 副本与评分阶段的多份副本内存减半，避免 analysis OOM / 缓存写不出导致的死亡循环。
    fd = list(full_dataset)
    fd[0] = np.ascontiguousarray(fd[0], dtype=np.float32)
    full_dataset = tuple(fd)
    if cache_path:
        save_full_dataset_to_parquet(cache_path, full_dataset)
        print(f"[dataset] 已缓存到: {cache_path}")
    return full_dataset


def make_synthetic_dataset(model, n_samples: int = 20000, seed: int = 0):
    """构造与真实模型特征维度一致的合成 full_dataset，仅用于无 DB 环境的代码冒烟测试。

    注: 合成数据不具经济含义，仅保证各分析模块可端到端运行。
    """
    rng = np.random.default_rng(seed)
    factor_names = list(model.feature_names)
    n = len(factor_names)
    # 横截面归一化后特征应落在 [0,1] 附近；以 0.5 为中心造数
    X = rng.uniform(0.0, 1.0, size=(n_samples, n)).astype(np.float32)
    dates = (
        pd.date_range("2018-01-01", periods=max(2, n_samples // 200), freq="B")
        .astype(str)
        .tolist()
    )
    date_arr = np.array(dates * ((n_samples // len(dates)) + 1))[:n_samples]
    rng.shuffle(date_arr)
    returns = rng.normal(0.0, 0.02, size=n_samples).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    unbuyable = np.zeros(n_samples, dtype=bool)
    limit_groups = np.zeros(n_samples, dtype=np.int64)
    path_scores = rng.uniform(0, 1, size=n_samples).astype(np.float32)
    is_st = np.zeros(n_samples, dtype=np.int8)
    w_sig = returns.copy()
    codes = np.array([f"{i:06d}" for i in rng.integers(1, 4000, size=n_samples)])
    return (
        X, y, returns, factor_names, date_arr, unbuyable,
        limit_groups, path_scores, is_st, w_sig, codes,
    )


# ════════════════════════════════════════════════════════════════════════
# 2. 归一化与预测
# ════════════════════════════════════════════════════════════════════════
def normalize_features(trainer, X: np.ndarray, dates: np.ndarray,
                       factor_names: List[str]) -> np.ndarray:
    """对特征做与训练完全一致的横截面归一化（原位修改副本）。"""
    Xn = np.array(X, dtype=np.float32, copy=True)
    trainer._apply_cross_sectional_normalization_inplace(Xn, np.asarray(dates), list(factor_names))
    return Xn


def predict_scores(model, Xn: np.ndarray, factor_names: List[str],
                   chunk_size: int = 200_000) -> np.ndarray:
    """用多目标模型对归一化特征打分（加权总分）。

    分块预测：避免一次性把全量特征矩阵 (178万×829) 包装成 float64 DataFrame
    造成 ~19GB 内存峰值（analysis OOM 主因之一）。每块仍按 pandas 默认升 float64，
    数值与原始全量预测逐元素一致。
    """
    factor_names = list(factor_names)
    n = Xn.shape[0]
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Xc = pd.DataFrame(np.ascontiguousarray(Xn[start:end]), columns=factor_names)
        chunks.append(np.asarray(model.predict(Xc), dtype=np.float32))
        del Xc
    return np.concatenate(chunks, axis=0)


# ════════════════════════════════════════════════════════════════════════
# 3. 特征归类: 财务 / 交易 / 估值 / 其他
# ════════════════════════════════════════════════════════════════════════
# 估值类特征（显式名单 + 命名模式）
_VALUATION_EXACT = {
    "dynamic_pe", "dynamic_pb", "inv_pe", "inv_pb", "market_cap",
    "peg", "roe_to_pb", "ep_ttm", "bp_ttm", "ps_ttm", "valuation_z",
}
_VALUATION_PREFIX = ("pe_", "pb_", "cap_", "ps_", "ev_", "pcf_")
_VALUATION_SUBSTR = ("_pe", "_pb", "_cap", "_peg", "valuation", "market_cap")

# 基础财务（非估值）显式名单
_FINANCIAL_EXACT = {
    "epsTTM", "roe", "roa", "roic", "liabilityToAsset", "assetToEquity",
    "profit_yoy", "revenue_yoy", "net_profit_yoy", "sue", "eav",
    "gross_margin", "operating_margin", "debt_to_asset", "current_ratio",
    "eps", "bvps", "ocfps", "roe_x_np_growth", "np_growth",
}
_FINANCIAL_PREFIX = ("roe_", "roa_", "eps", "profit", "revenue", "margin_",
                     "asset_", "debt_", "growth_", "sue", "eav", "equity_")


def _is_valuation(name: str) -> bool:
    if name in _VALUATION_EXACT:
        return True
    if any(name.startswith(p) for p in _VALUATION_PREFIX):
        return True
    if any(s in name for s in _VALUATION_SUBSTR):
        return True
    return False


def _is_financial(name: str) -> bool:
    if name in _FINANCIAL_EXACT:
        return True
    if any(name.startswith(p) for p in _FINANCIAL_PREFIX):
        return True
    return False


def classify_features(factor_names: List[str]) -> Dict[str, str]:
    """把每个特征名映射到四大类之一: 财务 / 交易 / 估值 / 其他。

    归类规则（与 prepare_dataset 的特征审计一致，并进一步拆分基本面为财务/估值）:
      - 估值: 市盈率/市净率/市值/PEG 等估值类
      - 财务: 盈利/成长/偿债/质量等基本面（非估值）
      - 交易: 技术面、量价、动量、波动、K线形态、市场情绪、状态、特征工程衍生
      - 其他: 以上均未匹配
    """
    cat: Dict[str, str] = {}
    for name in factor_names:
        if _is_valuation(name):
            cat[name] = "估值"
        elif _is_financial(name):
            cat[name] = "财务"
        else:
            # 交易面特征库（与审计中的 technical/advanced/candle/sentiment/status）
            cat[name] = "交易"
    return cat


def category_of_name(name: str) -> str:
    if _is_valuation(name):
        return "估值"
    if _is_financial(name):
        return "财务"
    return "交易"


# ════════════════════════════════════════════════════════════════════════
# 4. 牛 / 熊 / 震荡 市场状态划分
# ════════════════════════════════════════════════════════════════════════
def classify_regimes(
    dates: np.ndarray, returns: np.ndarray,
    fast: int = 20, slow: int = 60,
) -> Tuple[np.ndarray, pd.Series, Dict[str, str]]:
    """以等权截面收益作为市场代理，按快/慢均线趋势划分牛/熊/震荡。

    返回:
        regime_per_sample: 与样本对齐的状态数组 ('bull'/'bear'/'sideways')
        market_daily_ret:   市场每日等权收益序列（按日期排序）
        date_to_regime:     日期 -> 状态 映射
    """
    df = pd.DataFrame({"date": pd.Series(dates).astype(str).str[:10],
                       "ret": np.asarray(returns, dtype=float)})
    daily = df.groupby("date")["ret"].mean().sort_index()
    ma_fast = daily.rolling(fast, min_periods=max(5, fast // 2)).mean()
    ma_slow = daily.rolling(slow, min_periods=max(10, slow // 2)).mean()

    date_to_regime: Dict[str, str] = {}
    for d in daily.index:
        mf, ms = ma_fast.get(d, np.nan), ma_slow.get(d, np.nan)
        if pd.isna(mf) or pd.isna(ms) or ms == 0:
            date_to_regime[d] = "sideways"
        elif mf > ms and ms > 0:
            date_to_regime[d] = "bull"
        elif mf < ms and ms < 0:
            date_to_regime[d] = "bear"
        else:
            date_to_regime[d] = "sideways"

    regime_per_sample = np.array(
        [date_to_regime.get(str(d)[:10], "sideways") for d in dates], dtype=object
    )
    return regime_per_sample, daily, date_to_regime
