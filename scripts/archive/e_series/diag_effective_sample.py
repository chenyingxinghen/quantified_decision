"""量化「重叠滚动窗口的有效样本量」到底是多少。

背景：T046 用 se(rho)≈1/sqrt(N/h) 判定各族 regime 可预测性的显著性，
被质疑「有 13-17 年数据，且目标本来就是滚动窗口，为什么要除以 h」。

澄清两件不同的事：
  1. 滚动窗口确实产出 N-h+1 个样本（没有丢数据）；
  2. 但这些样本的目标区间重叠 h-1 天 → 强自相关 → 独立信息远少于 N。

本脚本不使用 N/h 这个经验公式，而是三种实测口径：
  A. 目标序列的自相关函数（ACF）与积分自相关时间 tau_int；
  B. Newey-West(HAC) 修正下 corr 的标准误；
  C. Block bootstrap 直接重采样出 corr 的抽样分布。

同时回答「用满 17 年、把 B 段扩大」能把有效样本量提到多少。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from datetime import datetime, timedelta  # noqa: E402

from config.baostock_config import DATABASE_PATH  # noqa: E402
from config.factor_config import TrainingConfig  # noqa: E402
from core.data.baostock_main import BaostockDataManager  # noqa: E402
from core.factors.nam_gate_model import NAMGateModel  # noqa: E402
from core.factors.regime_features import build_regime_matrix  # noqa: E402
from core.factors.train_ml_model import MLModelTrainer  # noqa: E402
from scripts.exp.exp_nam_gate import _prepare_fold  # noqa: E402
from scripts.archive.e_series.diag_regime_conditionality import (  # noqa: E402
    collect_group_panel,
    ridge_fit,
    ridge_pred,
)


def build_fold(model, n_stocks, years, end, workers=8):
    """复用 diag_regime_conditionality 的数据构建流程。"""
    end_dt = datetime.strptime(end, '%Y-%m-%d')
    end_s = end_dt.strftime('%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    TrainingConfig.FUTURE_DAYS = 7
    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()['code'].tolist()[:n_stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end_s)
    dataset = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end_s,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=workers, target_features=None, use_factor_cache_only=True,
        return_sample_metadata=False,
    )
    regime = build_regime_matrix(DATABASE_PATH)
    return _prepare_fold(trainer, dataset, list(model.feature_names),
                         0.8, 1.0, regime, target='returns')


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """样本自相关函数（已去均值、归一化）。"""
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    denom = float(np.dot(x, x))
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        out[k] = float(np.dot(x[: n - k], x[k:])) / denom if denom > 0 else np.nan
    return out


def tau_int(a: np.ndarray) -> float:
    """积分自相关时间：tau = 1 + 2*sum_{k>=1} rho_k，在 rho 首次<=0 处截断。

    有效样本量 n_eff = n / tau。这是 MCMC/时间序列里的标准口径，
    比 n/h 更贴近真实（它由数据决定，而非由构造 horizon 假定）。
    """
    s = 1.0
    for k in range(1, len(a)):
        if a[k] <= 0:
            break
        s += 2.0 * a[k]
    return max(s, 1.0)


def nw_se_corr(u: np.ndarray, v: np.ndarray, lag: int) -> tuple:
    """corr(u,v) 的 Newey-West(HAC) 标准误。

    把 corr 看作 z_u*z_v 的均值，对该乘积序列做 HAC 方差估计。
    """
    m = np.isfinite(u) & np.isfinite(v)
    u, v = u[m], v[m]
    n = len(u)
    if n < 30:
        return np.nan, np.nan
    zu = (u - u.mean()) / (u.std() + 1e-12)
    zv = (v - v.mean()) / (v.std() + 1e-12)
    g = zu * zv
    r = float(g.mean())
    gc = g - r
    var = float(np.dot(gc, gc)) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        c = float(np.dot(gc[: n - k], gc[k:])) / n
        var += 2.0 * w * c
    se = np.sqrt(max(var, 0.0) / n)
    return r, se


def block_bootstrap_se(u: np.ndarray, v: np.ndarray, block: int, n_boot: int = 2000,
                       seed: int = 7) -> float:
    """移动块自助法：保留块内时间依赖，直接重采样 corr 的抽样分布。"""
    m = np.isfinite(u) & np.isfinite(v)
    u, v = u[m], v[m]
    n = len(u)
    if n < block * 3:
        return np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts_pool = n - block
    out = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, starts_pool, size=nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        uu, vv = u[idx], v[idx]
        out[b] = np.corrcoef(uu, vv)[0, 1]
    return float(np.nanstd(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/nam_gate/T043/nam_gate_factor_model.pkl')
    ap.add_argument('--stocks', type=int, default=800)
    ap.add_argument('--years', type=int, default=17)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--out', default='diagnose_output/nam_gate/effective_sample.json')
    args = ap.parse_args()

    model = NAMGateModel()
    model.load_model(args.model)
    K = len(model.group_names)

    print(f"{'='*74}\n加载 {args.years} 年数据（此前实验只用了 13 年）\n{'='*74}")
    fold = build_fold(model, args.stocks, args.years, args.end)
    tr = collect_group_panel(model, fold['X_train'], fold['ret_train'],
                             fold['d_train'], fold['M_train'])
    va = collect_group_panel(model, fold['X_val'], fold['ret_val'],
                             fold['d_val'], fold['M_val'])
    gic = np.vstack([tr['group_ic'], va['group_ic']])
    dates = list(tr['dates']) + list(va['dates'])
    N = len(dates)
    print(f"  交易日总数 N = {N}   区间 {dates[0]} → {dates[-1]}")

    H = args.horizon
    df = pd.DataFrame(gic)
    Y = df.shift(-1).rolling(H, min_periods=max(3, H // 3)).mean().shift(-(H - 1)).to_numpy()
    n_valid = int(np.isfinite(Y).all(axis=1).sum())
    print(f"  滚动窗口产出的目标样本数 = {n_valid}（≈N，确实没丢数据）")

    # ---------------- A. 自相关 / 积分自相关时间 ----------------
    print(f"\n{'='*74}\n[A] 目标序列自相关 → 独立信息有多少\n{'='*74}")
    print(f"{'族':<12}{'rho(1)':>9}{'rho(10)':>9}{'rho(20)':>9}{'rho(40)':>9}"
          f"{'tau_int':>9}{'n_eff':>8}")
    rows = []
    for k, g in enumerate(model.group_names):
        y = Y[:, k]
        a = acf(y[np.isfinite(y)], nlags=80)
        t = tau_int(a)
        ne = n_valid / t
        rows.append({'group': g, 'acf1': a[1], 'acf10': a[10], 'acf20': a[20],
                     'acf40': a[40], 'tau_int': t, 'n_eff': ne})
        print(f"{g:<12}{a[1]:>9.3f}{a[10]:>9.3f}{a[20]:>9.3f}{a[40]:>9.3f}"
              f"{t:>9.1f}{ne:>8.0f}")
    tau_med = float(np.median([r['tau_int'] for r in rows]))
    print(f"\n  中位 tau_int = {tau_med:.1f}（若无重叠应 ≈1；构造 horizon h={H}）")
    print(f"  → 实测有效样本量 ≈ {n_valid/tau_med:.0f}，而非名义 {n_valid}")

    # ---------------- B. 严格样本外预测 + HAC 标准误 ----------------
    print(f"\n{'='*74}\n[B] 用满数据重做 OOS 预测，HAC 修正标准误\n{'='*74}")
    nA = int(N * 0.6)
    okA = np.isfinite(Y[:nA]).all(axis=1)
    M = np.vstack([
        np.array([fold['M_train'][{d: i for i, d in enumerate(fold['d_train'])}[d]]
                  for d in tr['dates']], dtype=np.float64),
        np.array([fold['M_val'][{d: i for i, d in enumerate(fold['d_val'])}[d]]
                  for d in va['dates']], dtype=np.float64),
    ])
    rg = ridge_fit(M[:nA][okA], Y[:nA][okA], alpha=50.0)
    P = ridge_pred(rg, M[nA:])
    Yo = Y[nA:]
    n_oos = int(np.isfinite(Yo).all(axis=1).sum())
    print(f"  OOS 段样本 {n_oos} 天（此前实验的 B 段只有 496 天）")
    print(f"{'族':<12}{'corr':>8}{'naive se':>10}{'HAC se':>9}{'boot se':>9}"
          f"{'|t|_HAC':>9}  判定")
    outB = []
    for k, g in enumerate(model.group_names):
        r, se_hac = nw_se_corr(P[:, k], Yo[:, k], lag=int(3 * H))
        se_naive = 1.0 / np.sqrt(max(n_oos, 2))
        se_boot = block_bootstrap_se(P[:, k], Yo[:, k], block=int(2 * H))
        t = abs(r) / se_hac if se_hac and np.isfinite(se_hac) else np.nan
        verdict = '可信' if t > 2 else ('边缘' if t > 1 else '不可分辨于0')
        outB.append({'group': g, 'corr': r, 'se_naive': se_naive,
                     'se_hac': se_hac, 'se_boot': se_boot, 't_hac': t})
        print(f"{g:<12}{r:>+8.3f}{se_naive:>10.3f}{se_hac:>9.3f}{se_boot:>9.3f}"
              f"{t:>9.2f}  {verdict}")

    se_hac_med = float(np.nanmedian([r['se_hac'] for r in outB]))
    se_naive0 = 1.0 / np.sqrt(max(n_oos, 2))
    print(f"\n  朴素 se（当每天独立）= {se_naive0:.3f}")
    print(f"  HAC  se（实测）       = {se_hac_med:.3f}   膨胀 {se_hac_med/se_naive0:.1f}x")
    print(f"  → 等效独立样本 ≈ {1.0/se_hac_med**2:.0f}（名义 {n_oos}）")

    # ---------------- C. 更多数据能提升多少 ----------------
    print(f"\n{'='*74}\n[C] 加数据 vs 缩 horizon：哪个更划算\n{'='*74}")
    print(f"  当前 OOS {n_oos} 天 / tau≈{tau_med:.0f} → n_eff ≈ {n_oos/tau_med:.0f}")
    print(f"  若 OOS 扩到 17 年全量（≈4100 天）→ n_eff ≈ {4100/tau_med:.0f}"
          f"   se ≈ {1/np.sqrt(4100/tau_med):.3f}")
    print(f"  若 horizon 20→5（tau 同比例降至 ≈{tau_med/4:.0f}）")
    print(f"    当前样本下 n_eff ≈ {n_oos/(tau_med/4):.0f}"
          f"   se ≈ {1/np.sqrt(n_oos/(tau_med/4)):.3f}")
    print(f"  注：se ∝ 1/sqrt(n_eff)，样本翻倍只降 se 到 0.71x；"
          f"horizon 减半直接把 n_eff 翻倍，且不需要新数据。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({
            'n_days': N, 'date_range': [dates[0], dates[-1]],
            'horizon': H, 'n_target_samples': n_valid,
            'acf_table': rows, 'tau_int_median': tau_med,
            'oos_days': n_oos, 'oos_table': outB,
            'se_naive': se_naive0, 'se_hac_median': se_hac_med,
        }, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n已保存 {args.out}")


if __name__ == '__main__':
    main()
