"""
优化实验：双模型集成 vs 单模型，针对"验证 IC 不稳定"（std≈0.20，均值≈0.05）这一真实症状。

思路：ranker 单模型的日间 IC 方差很大。两个不同模型（xgb 连续 rank / lgb 离散 lambdarank）
      的误差不完全相关，等权平均预测应能降低 IC 的日间标准差，即使均值 IC 不涨，
      IC_IR = mean/std 提升本身就是可交易性的改进。

复用现有 train_models 流程（会就地归一化传入的 X），训练后手动切出验证集，
对 xgb / lgb / ensemble 三者用同一套逐日 Rank IC 评估，输出 mean / std / IR / top1 / win。

用法:
    EXP_STOCKS=3000 EXP_YEARS=15 python scripts/exp_ensemble.py
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import spearmanr

from core.factors.train_ml_model import MLModelTrainer
from config.baostock_config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager


def eval_preds(y_prob, returns, dates, min_n=10):
    """逐日截面 Rank IC + Top-1/Top-5/胜率。返回 dict。"""
    ics, top1_hits, top5_hits, wins = [], [], [], []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < min_n:
            continue
        p, r = y_prob[m], returns[m]
        if len(np.unique(p)) > 1 and len(np.unique(r)) > 1:
            ic, _ = spearmanr(p, r)
            if not np.isnan(ic):
                ics.append(ic)
        if len(r) >= 10:
            t1 = np.argmax(p)
            top1_hits.append(1.0 if r[t1] >= np.percentile(r, 95) else 0.0)
            wins.append(1.0 if r[t1] > 0 else 0.0)
            n_top = min(5, len(r))
            t5 = np.argsort(p)[-n_top:]
            top5_hits.append(float(np.mean(r[t5] >= np.percentile(r, 80))))
    mean_ic = float(np.mean(ics)) if ics else 0.0
    std_ic = float(np.std(ics)) if ics else 0.0
    return {
        'ic': mean_ic, 'ic_std': std_ic,
        'ir': mean_ic / (std_ic + 1e-9),
        'top1': float(np.mean(top1_hits)) if top1_hits else 0.0,
        'top5': float(np.mean(top5_hits)) if top5_hits else 0.0,
        'win': float(np.mean(wins)) if wins else 0.0,
        'n_days': len(ics),
    }


def compute_val_slice(dates):
    """复刻 train_models 的带 embargo 时间划分，返回 val_start_idx。"""
    forward_days = getattr(TrainingConfig, 'FUTURE_DAYS', 7)
    raw_split_idx = int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)
    split_date = dates[raw_split_idx]
    unique_dates = np.unique(dates)
    split_date_idx = np.searchsorted(unique_dates, split_date)
    val_start_date = unique_dates[min(split_date_idx + forward_days, len(unique_dates) - 1)]
    return np.searchsorted(dates, val_start_date, side='left')


def main():
    N_STOCKS = int(os.environ.get('EXP_STOCKS', '3000'))
    YEARS = int(os.environ.get('EXP_YEARS', '15'))
    end = (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)).strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365 * (TrainingConfig.YEARS_FOR_BACKTEST + YEARS))).strftime('%Y-%m-%d')

    print(f"=== 集成 vs 单模型 实验 ===")
    print(f"股票: {N_STOCKS}, 窗口: {start} ~ {end}")

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()['code'].tolist()[:N_STOCKS]
    manager.close()

    t0 = time.time()
    stocks_data = trainer.load_label_data(codes, start, end)
    print(f"标签行情加载: {len(stocks_data)} 只, {time.time()-t0:.1f}s")

    ds = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=True, n_jobs=15,
        target_features=None, use_factor_cache_only=True,
    )
    del stocks_data
    X, y, returns, factor_names, dates, unbuyable, limit_groups, path_scores, is_st, w_sig = ds

    # train_models 会就地归一化 X（train 段与 val 段分别归一化），训练后 X 即为归一化后的矩阵
    factor_names_orig = list(factor_names)
    trainer.train_models(
        X, y, returns, factor_names, dates,
        unbuyable_mask=unbuyable, limit_groups=limit_groups,
        path_scores=path_scores, is_st_arr=is_st, w_sig_arr=w_sig,
        model_types=['xgboost', 'lightgbm'],
    )

    # 手动切验证集（X 已被就地归一化）
    val_start = compute_val_slice(dates)
    X_val = X[val_start:]
    dates_val = dates[val_start:]
    returns_val = returns[val_start:]
    X_val_df = pd.DataFrame(X_val, columns=factor_names_orig)

    xgb_model = trainer.models.get('xgboost')
    lgb_model = trainer.models.get('lightgbm')
    if xgb_model is None or lgb_model is None:
        print("模型训练不完整，无法比较集成")
        return

    p_xgb = xgb_model.predict(X_val_df)
    p_lgb = lgb_model.predict(X_val_df)
    # 集成前把两模型输出各自转成截面排名，消除量纲差异，再等权平均
    def to_rank(p, dates):
        out = np.empty_like(p, dtype=np.float64)
        for d in np.unique(dates):
            m = dates == d
            v = p[m]
            r = v.argsort().argsort().astype(np.float64)
            out[m] = r / (len(v) + 1) if len(v) > 1 else 0.5
        return out
    p_ens = 0.5 * to_rank(p_xgb, dates_val) + 0.5 * to_rank(p_lgb, dates_val)

    r_xgb = eval_preds(p_xgb, returns_val, dates_val)
    r_lgb = eval_preds(p_lgb, returns_val, dates_val)
    r_ens = eval_preds(p_ens, returns_val, dates_val)

    print(f"\n\n{'='*78}\n验证集对比 (集成针对 IC 不稳定症状; IR=IC均值/IC标准差)\n{'='*78}")
    print(f"{'model':>10} | {'IC':>7} | {'IC_std':>7} | {'IR':>6} | {'top1':>6} | {'top5':>6} | {'win':>6}")
    print('-'*78)
    for name, r in [('xgboost', r_xgb), ('lightgbm', r_lgb), ('ENSEMBLE', r_ens)]:
        print(f"{name:>10} | {r['ic']:>7.4f} | {r['ic_std']:>7.4f} | {r['ir']:>6.3f} | "
              f"{r['top1']:>6.2%} | {r['top5']:>6.2%} | {r['win']:>6.2%}")
    print('='*78)
    print(f"验证交易日数: {r_ens['n_days']}")


if __name__ == '__main__':
    main()
