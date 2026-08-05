"""
快速诊断实验：用小样本验证标签质量、特征信号强度、过拟合来源。
只读缓存，不写模型。用于迭代排查，不进入正式训练流程。
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core.factors.train_ml_model import MLModelTrainer
from config.baostock_config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager


def cross_sectional_ic(pred, ref, dates, min_n=10):
    """按日期分组算 Rank IC 均值。"""
    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < min_n:
            continue
        a, b = pred[m], ref[m]
        if len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
            ic, _ = spearmanr(a, b)
            if not np.isnan(ic):
                ics.append(ic)
    return np.mean(ics) if ics else 0.0, np.std(ics) if ics else 0.0, len(ics)


def main():
    N_STOCKS = int(os.environ.get('EXP_STOCKS', '800'))
    YEARS = int(os.environ.get('EXP_YEARS', '6'))
    from datetime import datetime, timedelta
    end = (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)).strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365 * (TrainingConfig.YEARS_FOR_BACKTEST + YEARS))).strftime('%Y-%m-%d')

    print(f"=== 诊断实验 ===\n股票: {N_STOCKS}, 窗口: {start} ~ {end}")

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()['code'].tolist()[:N_STOCKS]
    manager.close()

    t0 = time.time()
    stocks_data = trainer.load_label_data(codes, start, end)
    print(f"标签行情加载: {len(stocks_data)} 只, {time.time()-t0:.1f}s")

    t0 = time.time()
    ds = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=True, n_jobs=15,
        target_features=None, use_factor_cache_only=True,
    )
    del stocks_data
    X, y, returns, factor_names, dates, unbuyable, limit_groups, path_scores, is_st, w_sig = ds
    print(f"prepare_dataset: {time.time()-t0:.1f}s | X={X.shape}, 特征数={len(factor_names)}")

    # ---- 1. 标签自身 vs 收益率 的截面 Rank IC（验证"严格单调"）----
    print("\n[1] 标签质量：标签 y (截面rank) vs 真实收益 returns")
    ic, std, nd = cross_sectional_ic(y, returns, dates)
    print(f"    label-return Rank IC = {ic:.4f} ± {std:.4f}  ({nd} 天)")
    print(f"    -> 若接近 1，说明标签与收益严格单调；若明显 <1，说明 path/vol 项打破了单调")
    ic_ps, _, _ = cross_sectional_ic(path_scores, returns, dates)
    print(f"    path_score-return Rank IC = {ic_ps:.4f}  (原始分数 vs 收益)")

    # ---- 2. 特征单变量信号强度：每个特征 vs 收益 的截面 IC ----
    print("\n[2] 特征信号强度：单变量截面 IC (归一化前，取样加速)")
    # 抽样降低耗时
    sample_days = np.random.default_rng(0).choice(np.unique(dates),
                    size=min(200, len(np.unique(dates))), replace=False)
    smask = np.isin(dates, sample_days)
    Xs, rs, ds_s = X[smask], returns[smask], dates[smask]
    feat_ic = []
    for j, name in enumerate(factor_names):
        ic_j, _, _ = cross_sectional_ic(Xs[:, j], rs, ds_s)
        feat_ic.append((name, ic_j))
    feat_ic.sort(key=lambda t: abs(t[1]), reverse=True)
    print("    Top-15 |IC| 特征:")
    for name, v in feat_ic[:15]:
        print(f"      {name:40s} IC={v:+.4f}")
    strong = [t for t in feat_ic if abs(t[1]) > 0.02]
    print(f"    |IC|>0.02 的特征数: {len(strong)} / {len(factor_names)}")

    print("\n=== 诊断完成 ===")


if __name__ == '__main__':
    main()
