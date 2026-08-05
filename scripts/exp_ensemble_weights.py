"""
优化实验：集成【权重搜索】。当前生产集成是等权 0.5/0.5，但 lgb 单模型 IR 更高，
偏向 lgb 的加权可能给出更好的 IR。本脚本在验证集上扫描 xgb 权重 w ∈ [0,1]，
对每个 w 用 w*rank(xgb)+(1-w)*rank(lgb) 融合，复用 exp_ensemble 的逐日 Rank IC 评估。

关键：训练一次后，val 上的 p_xgb/p_lgb 固定，扫描权重几乎零成本——无需重训。
选出验证 IR 最高的权重，若显著优于等权，再用它重建生产集成 pkl。

用法:
    EXP_STOCKS=3000 EXP_YEARS=15 python -u scripts/exp_ensemble_weights.py
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
    ics, top1_hits, wins = [], [], []
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
    mean_ic = float(np.mean(ics)) if ics else 0.0
    std_ic = float(np.std(ics)) if ics else 0.0
    return {
        'ic': mean_ic, 'ic_std': std_ic,
        'ir': mean_ic / (std_ic + 1e-9),
        'top1': float(np.mean(top1_hits)) if top1_hits else 0.0,
        'win': float(np.mean(wins)) if wins else 0.0,
        'n_days': len(ics),
    }


def compute_val_slice(dates):
    forward_days = getattr(TrainingConfig, 'FUTURE_DAYS', 7)
    raw_split_idx = int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)
    split_date = dates[raw_split_idx]
    unique_dates = np.unique(dates)
    split_date_idx = np.searchsorted(unique_dates, split_date)
    val_start_date = unique_dates[min(split_date_idx + forward_days, len(unique_dates) - 1)]
    return np.searchsorted(dates, val_start_date, side='left')


def to_rank(p, dates):
    out = np.empty_like(p, dtype=np.float64)
    for d in np.unique(dates):
        m = dates == d
        v = p[m]
        out[m] = v.argsort().argsort().astype(np.float64) / (len(v) + 1) if len(v) > 1 else 0.5
    return out


def main():
    N_STOCKS = int(os.environ.get('EXP_STOCKS', '3000'))
    YEARS = int(os.environ.get('EXP_YEARS', '15'))
    end = (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)).strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365 * (TrainingConfig.YEARS_FOR_BACKTEST + YEARS))).strftime('%Y-%m-%d')

    print(f"=== 集成权重搜索 实验 ===")
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

    factor_names_orig = list(factor_names)
    trainer.train_models(
        X, y, returns, factor_names, dates,
        unbuyable_mask=unbuyable, limit_groups=limit_groups,
        path_scores=path_scores, is_st_arr=is_st, w_sig_arr=w_sig,
        model_types=['xgboost', 'lightgbm'],
    )

    val_start = compute_val_slice(dates)
    X_val = X[val_start:]
    dates_val = dates[val_start:]
    returns_val = returns[val_start:]
    X_val_df = pd.DataFrame(X_val, columns=factor_names_orig)

    xgb_model = trainer.models.get('xgboost')
    lgb_model = trainer.models.get('lightgbm')
    if xgb_model is None or lgb_model is None:
        print("模型训练不完整，无法搜索权重")
        return

    p_xgb = np.asarray(xgb_model.predict(X_val_df), dtype=np.float64)
    p_lgb = np.asarray(lgb_model.predict(X_val_df), dtype=np.float64)
    rank_xgb = to_rank(p_xgb, dates_val)
    rank_lgb = to_rank(p_lgb, dates_val)

    # 单模型基准
    r_xgb = eval_preds(p_xgb, returns_val, dates_val)
    r_lgb = eval_preds(p_lgb, returns_val, dates_val)

    print(f"\n{'='*82}")
    print(f"单模型基准 | xgb IR={r_xgb['ir']:.3f} (IC={r_xgb['ic']:.4f})  "
          f"lgb IR={r_lgb['ir']:.3f} (IC={r_lgb['ic']:.4f})")
    print(f"{'='*82}")
    print(f"{'w_xgb':>6} | {'w_lgb':>6} | {'IC':>7} | {'IC_std':>7} | {'IR':>6} | {'top1':>6} | {'win':>6}")
    print('-'*82)

    results = []
    for w in np.round(np.arange(0.0, 1.001, 0.1), 2):
        p_ens = w * rank_xgb + (1.0 - w) * rank_lgb
        r = eval_preds(p_ens, returns_val, dates_val)
        results.append((w, r))
        tag = '  <-- 当前生产(等权)' if abs(w - 0.5) < 1e-6 else ''
        print(f"{w:>6.1f} | {1-w:>6.1f} | {r['ic']:>7.4f} | {r['ic_std']:>7.4f} | "
              f"{r['ir']:>6.3f} | {r['top1']:>6.2%} | {r['win']:>6.2%}{tag}")
    print('='*82)

    best_w, best_r = max(results, key=lambda t: t[1]['ir'])
    eq_r = [r for w, r in results if abs(w - 0.5) < 1e-6][0]
    print(f"\n最优权重(按IR): w_xgb={best_w:.1f} / w_lgb={1-best_w:.1f}  IR={best_r['ir']:.3f}")
    print(f"等权基准:      w_xgb=0.5 / w_lgb=0.5           IR={eq_r['ir']:.3f}")
    if best_r['ir'] > eq_r['ir']:
        gain = (best_r['ir'] - eq_r['ir']) / (abs(eq_r['ir']) + 1e-9) * 100
        print(f"最优权重相对等权 IR 提升 {gain:+.1f}%  ->  值得用 w_xgb={best_w:.1f} 重建生产集成")
    else:
        print(f"等权已是(近似)最优，无需调整权重")
    print(f"验证交易日数: {best_r['n_days']}")


if __name__ == '__main__':
    main()
