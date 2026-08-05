"""
标签分解诊断：定位"训练IC高、验证IC低"的元凶是否出在标签设计。

模型训练用的标签是复合"路径质量分"：
    base_score  = f_returns_norm*3 + upside*2 + downside*1
    final_score = base_score * vol_booster * path_mult
    vol_booster = 1 + rel_atr * VOL_BOOSTER_COEF   (默认 10)
    path_mult   = 1 ± path_bonus/path_penalty

若某个乘子把标签推离真实未来收益，模型就在优化错误目标——训练集能拟合
这个被扭曲的标签，但拿到真实收益上就崩。本脚本逐步测每一层与真实未来收益
f_returns_raw 的逐日截面 Rank IC，哪一步 IC 下降即元凶。

关键：挂钩 _calculate_vectorized_labels（主进程单次调用，含完整 date 列与全部
原始输入），在其中复现各层分数，不重训、只读缓存。

用法:
    EXP_STOCKS=800 EXP_YEARS=6 python -u scripts/exp_label_decomp.py
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


def xs_ic(pred, ref, dates, min_n=10):
    """逐日截面 Rank IC 均值/标准差/天数。"""
    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < min_n:
            continue
        a, b = pred[m], ref[m]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) >= min_n and len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
            ic, _ = spearmanr(a, b)
            if not np.isnan(ic):
                ics.append(ic)
    if not ics:
        return 0.0, 0.0, 0
    return float(np.mean(ics)), float(np.std(ics)), len(ics)


# 全局捕获槽
_CAP = {}


def _install_hook(trainer):
    """挂钩 _calculate_vectorized_labels，抓取 components 与各层分数。"""
    orig = trainer._calculate_vectorized_labels

    def wrapped(components: pd.DataFrame):
        eps = 1e-8
        next_open = components['next_open'].values
        limits = components['limit_thresholds'].values

        f_returns_raw = (components['f_close'].values / (next_open + eps)) - 1
        f_high_max_raw = (components['f_high_max'].values / (next_open + eps)) - 1
        f_low_min_raw = (components['f_low_min'].values / (next_open + eps)) - 1

        f_returns_norm = f_returns_raw / limits
        f_high_max_norm = f_high_max_raw / limits
        f_low_min_norm = f_low_min_raw / limits

        upside = np.where(f_high_max_norm > 0, f_high_max_norm, 0)
        downside = np.where(f_low_min_norm < 0, f_low_min_norm, 0)

        w_upside = getattr(TrainingConfig, 'UPSIDE_WEIGHT', 2.0)
        w_downside = getattr(TrainingConfig, 'DOWNSIDE_WEIGHT', 1.0)
        w_final = getattr(TrainingConfig, 'FINAL_RETURN_WEIGHT', 3.0)
        base_score = (f_returns_norm * w_final) + (upside * w_upside) + (downside * w_downside)

        _vol_coef = getattr(TrainingConfig, 'VOL_BOOSTER_COEF', 10.0)
        rel_atr = components['atr_rel'].values
        vol_booster = 1.0 + (rel_atr * _vol_coef)

        high_idx = np.asarray(components['f_high_idx'].values, dtype=np.float64)
        low_idx = np.asarray(components['f_low_idx'].values, dtype=np.float64)
        path_bonus = getattr(TrainingConfig, 'PATH_BONUS', 0.15)
        path_penalty = getattr(TrainingConfig, 'PATH_PENALTY', 0.10)
        path_mult = np.where(
            np.isnan(high_idx) | np.isnan(low_idx), 1.0,
            np.where(high_idx < low_idx, 1.0 + path_bonus,
                     np.where(low_idx < high_idx, 1.0 - path_penalty, 1.0)))

        _CAP['dates'] = components['date'].values.copy()
        _CAP['ret'] = f_returns_raw.copy()          # 真实未来收益（裁判）
        _CAP['ret_norm'] = f_returns_norm.copy()    # 除涨跌停后的收益
        _CAP['base'] = base_score.copy()
        _CAP['vol_booster'] = vol_booster.copy()
        _CAP['rel_atr'] = rel_atr.copy()
        _CAP['path_mult'] = path_mult.copy()
        _CAP['base_vol'] = (base_score * vol_booster).copy()
        _CAP['final'] = (base_score * vol_booster * path_mult).copy()
        _CAP['upside'] = upside.copy()
        _CAP['downside'] = downside.copy()

        return orig(components)

    trainer._calculate_vectorized_labels = wrapped


def main():
    N_STOCKS = int(os.environ.get('EXP_STOCKS', '800'))
    YEARS = int(os.environ.get('EXP_YEARS', '6'))
    end = (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)).strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365 * (TrainingConfig.YEARS_FOR_BACKTEST + YEARS))).strftime('%Y-%m-%d')

    print(f"=== 标签分解诊断 ===")
    print(f"股票: {N_STOCKS}, 窗口: {start} ~ {end}")
    print(f"当前配置: FINAL_RETURN_WEIGHT={getattr(TrainingConfig,'FINAL_RETURN_WEIGHT',3.0)}, "
          f"UPSIDE_WEIGHT={getattr(TrainingConfig,'UPSIDE_WEIGHT',2.0)}, "
          f"DOWNSIDE_WEIGHT={getattr(TrainingConfig,'DOWNSIDE_WEIGHT',1.0)}, "
          f"VOL_BOOSTER_COEF={getattr(TrainingConfig,'VOL_BOOSTER_COEF',10.0)}, "
          f"PATH_BONUS={getattr(TrainingConfig,'PATH_BONUS',0.15)}, "
          f"PATH_PENALTY={getattr(TrainingConfig,'PATH_PENALTY',0.10)}")

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    _install_hook(trainer)

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

    if not _CAP:
        print("!! 未捕获到标签中间量，挂钩可能未命中")
        return

    dates = _CAP['dates']
    ret = _CAP['ret']

    print(f"\n捕获样本量: {len(ret)}")
    print(f"\n{'='*72}")
    print("各层标签 vs 真实未来收益 f_returns_raw 的逐日截面 Rank IC")
    print("（IC 越高=越贴近真实收益；某一步 IC 下降=该乘子破坏单调性）")
    print(f"{'='*72}")
    print(f"{'标签层':<34} | {'RankIC':>8} | {'IC_std':>7} | {'天数':>5}")
    print('-'*72)

    layers = [
        ('ret_norm (收益/涨跌停, 基准上界)', 'ret_norm'),
        ('base_score (收益*3+上行*2+下行)', 'base'),
        ('base * vol_booster', 'base_vol'),
        ('base * vol * path (=最终标签)', 'final'),
    ]
    for label, key in layers:
        ic, std, nd = xs_ic(_CAP[key], ret, dates)
        print(f"{label:<34} | {ic:>8.4f} | {std:>7.4f} | {nd:>5}")

    print('-'*72)
    # 乘子本身与收益的关系（负相关=拉低标签质量）
    ic_vol, _, _ = xs_ic(_CAP['rel_atr'], ret, dates)
    ic_up, _, _ = xs_ic(_CAP['upside'], ret, dates)
    ic_dn, _, _ = xs_ic(_CAP['downside'], ret, dates)
    print(f"{'[单变量] rel_atr(波动) vs 收益':<34} | {ic_vol:>8.4f} |    -    |   -   "
          f"  <- 负值印证低波动异象")
    print(f"{'[单变量] upside(上行空间) vs 收益':<34} | {ic_up:>8.4f} |    -    |   -")
    print(f"{'[单变量] downside(下行) vs 收益':<34} | {ic_dn:>8.4f} |    -    |   -")
    print('='*72)

    # 结论提示
    ic_base, _, _ = xs_ic(_CAP['base'], ret, dates)
    ic_bv, _, _ = xs_ic(_CAP['base_vol'], ret, dates)
    ic_fin, _, _ = xs_ic(_CAP['final'], ret, dates)
    print("\n[结论]")
    print(f"  base -> ×vol : IC {ic_base:.4f} -> {ic_bv:.4f}  (Δ={ic_bv-ic_base:+.4f})")
    print(f"  ×vol -> ×path: IC {ic_bv:.4f} -> {ic_fin:.4f}  (Δ={ic_fin-ic_bv:+.4f})")
    if ic_bv < ic_base - 0.002:
        print("  => vol_booster 拉低了标签与真实收益的对齐度，是元凶之一。")
    if ic_fin < ic_bv - 0.002:
        print("  => path_mult 进一步拉低对齐度。")
    if ic_base >= ic_fin:
        print(f"  => 纯 base_score 的 IC({ic_base:.4f}) >= 最终标签({ic_fin:.4f})，"
              f"复合乘子非但无益反而有害。")


if __name__ == '__main__':
    main()
