"""
诊断：训练标签（scores）与真实前向收益（returns）在逐日截面上的相关性。

动机：T039 中 NAMGateModel 对标签的验证 Rank IC 达 +0.11，但对真实收益的
Rank IC 为 -0.04，而同数据 XGBoost 基线对真实收益为 +0.08。若标签本身与收益
的截面相关性偏低，说明标签中的非收益成分（波动率奖励等）足以让一个"忠实拟合
标签"的模型在真实收益上失效——这正是加性平滑模型（NAM）比 GBDT 更容易踩的坑。
"""

import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from datetime import datetime, timedelta  # noqa: E402

from config import DATABASE_PATH  # noqa: E402
from config.factor_config import TrainingConfig  # noqa: E402
from core.data.baostock_main import BaostockDataManager  # noqa: E402
from core.factors.train_ml_model import MLModelTrainer  # noqa: E402
from scripts.diagnose_xgb_oof_head import _fold_dataset  # noqa: E402


def daily_spearman(a, b, dates):
    out = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 5:
            continue
        r = spearmanr(a[m], b[m]).correlation
        if np.isfinite(r):
            out.append(r)
    return float(np.mean(out)), len(out)


def main():
    end_dt = datetime.strptime('2024-08-01', '%Y-%m-%d')
    end = end_dt.strftime('%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * 4)).strftime('%Y-%m-%d')
    TrainingConfig.FUTURE_DAYS = 7
    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()['code'].tolist()[:200]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)
    dataset = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=4, target_features=None, use_factor_cache_only=True,
        return_sample_metadata=False,
    )
    fold, _, val_start, _, split_date, end_date = _fold_dataset(dataset, 0.8, 1.0)
    X, y, returns, names, dates, *_rest = fold
    scores = fold[7]

    dv, rv, sv = dates[val_start:], returns[val_start:], scores[val_start:]
    yv = y[val_start:]

    print(f"验证段 {split_date} → {end_date}，{len(dv)} 样本")
    for label, arr in (('scores(训练标签)', sv), ('y(离散档位)', yv)):
        ic, n = daily_spearman(arr, rv, dv)
        print(f"  逐日截面 Rank IC({label} , 真实收益) = {ic:+.4f}  ({n} 日)")

    # 标签内部成分：与波动率的关系
    idx = {n: i for i, n in enumerate(names)}
    for vol_col in ('atr_14', 'volatility_20', 'relative_atr'):
        if vol_col not in idx:
            continue
        v = X[val_start:, idx[vol_col]]
        ic_s, _ = daily_spearman(sv, v, dv)
        ic_r, _ = daily_spearman(rv, v, dv)
        print(f"  {vol_col:16s} vs 标签 IC={ic_s:+.4f} | vs 真实收益 IC={ic_r:+.4f}")


if __name__ == '__main__':
    main()
