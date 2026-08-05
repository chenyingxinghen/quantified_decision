"""
对照实验：VOL_BOOSTER_COEF 对验证集 Rank IC 的影响。
诊断发现高波动股未来收益偏低（IC≈-0.07），怀疑正的 vol_booster 拉低模型表现。
本脚本在同规模数据上分别用 coef=10（基线）和 coef=0（关闭）训练 XGBoost，对比 train/val Rank IC。

用法:
    EXP_STOCKS=1500 EXP_YEARS=8 python scripts/exp_vol_booster.py
    EXP_COEFS=10,0,5 python scripts/exp_vol_booster.py   # 自定义对照系数
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta

from core.factors.train_ml_model import MLModelTrainer
from config.baostock_config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager


def run_once(trainer, stocks_data, start, end, coef, model_types):
    """用给定 vol_booster 系数重算标签并训练，返回 {model: (train_ic, val_ic)}"""
    TrainingConfig.VOL_BOOSTER_COEF = coef
    print(f"\n{'#'*70}\n# 实验: VOL_BOOSTER_COEF = {coef}\n{'#'*70}")

    ds = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=True, n_jobs=15,
        target_features=None, use_factor_cache_only=True,
    )
    X, y, returns, factor_names, dates, unbuyable, limit_groups, path_scores, is_st, w_sig = ds

    results = trainer.train_models(
        X, y, returns, factor_names, dates,
        unbuyable_mask=unbuyable, limit_groups=limit_groups,
        path_scores=path_scores, is_st_arr=is_st, w_sig_arr=w_sig,
        model_types=model_types,
    )
    out = {}
    for mt, r in results.items():
        tr = r.get('train_metrics', {}).get('rank_ic', float('nan'))
        va = r.get('val_metrics', {}).get('rank_ic', float('nan'))
        va_top1 = r.get('val_metrics', {}).get('top1_precision', float('nan'))
        va_win = r.get('val_metrics', {}).get('win_rate', float('nan'))
        out[mt] = (tr, va, va_top1, va_win)
    return out


def main():
    N_STOCKS = int(os.environ.get('EXP_STOCKS', '1500'))
    YEARS = int(os.environ.get('EXP_YEARS', '8'))
    coefs = [float(c) for c in os.environ.get('EXP_COEFS', '10,0').split(',')]
    model_types = os.environ.get('EXP_MODELS', 'xgboost').split(',')

    end = (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)).strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365 * (TrainingConfig.YEARS_FOR_BACKTEST + YEARS))).strftime('%Y-%m-%d')

    print(f"=== VOL_BOOSTER 对照实验 ===")
    print(f"股票: {N_STOCKS}, 窗口: {start} ~ {end}, 系数对照: {coefs}, 模型: {model_types}")

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()['code'].tolist()[:N_STOCKS]
    manager.close()

    t0 = time.time()
    stocks_data = trainer.load_label_data(codes, start, end)
    print(f"标签行情加载: {len(stocks_data)} 只, {time.time()-t0:.1f}s")

    all_results = {}
    for coef in coefs:
        all_results[coef] = run_once(trainer, stocks_data, start, end, coef, model_types)

    print(f"\n\n{'='*70}\n对照汇总 (Rank IC / Top1精度 / 胜率)\n{'='*70}")
    print(f"{'coef':>6} | {'model':>10} | {'train_ic':>9} | {'val_ic':>8} | {'top1':>7} | {'win':>7}")
    print('-'*70)
    for coef in coefs:
        for mt, (tr, va, t1, win) in all_results[coef].items():
            print(f"{coef:>6} | {mt:>10} | {tr:>9.4f} | {va:>8.4f} | {t1:>7.2%} | {win:>7.2%}")
    print('='*70)


if __name__ == '__main__':
    main()
