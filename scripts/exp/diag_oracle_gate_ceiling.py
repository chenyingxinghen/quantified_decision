#!/usr/bin/env python
"""T129：**族权重的 oracle 天花板** —— 到底是「没信息」还是「有信息但不可预测」？

问题
----
用户的直觉：regime 差异真实存在（T115 跌日 IC 0.135 vs 涨日 0.091、熊牛回测差 100+pp），
「就算样本少，if-else 也该有点帮助」。但 T116~T128 的门控臂全部 Δ≈0 或为负。
到底卡在哪？本脚本把它拆成三个可测的量，一次回答。

⚠ 先厘清一个**架构事实**：策略每天按分数排名取前 K 只，所以任何**只随日期变化、
  不改变当日股票间相对顺序**的调整（"今天熊市整体谨慎"）对选股**完全不可见** ——
  它只能影响仓位/空仓，那是风控规则（用户已禁止）。regime 信息要影响选股，
  **必须改变因子之间的相对权重**，这正是门控做的事。所以本测量的对象就是
  「逐日最优族权重」，它是**任何**形式的 regime 条件化（含 if-else）的共同上界。

三个量
------
1. **oracle 天花板**：允许用当天的**未来收益**去解每日最优权重 w*_d
   （岭回归闭式解），得到「线性族重加权」在该日能达到的 IC 上界。
   这一步回答「信息存在吗」—— 天花板远高于基线 ⇒ 存在。
2. **可预测份额**：w*_d 能被当日 regime 向量 m_d（30 维）预测多少？
   在训练段拟合、验证段报 **折外 R²**。这一步回答「学得到吗」。
3. **两个可实现方案的实测收益**（都只用过去信息）：
   a. `regime`   ：ŵ_d = 训练段拟合的 m_d → w 线性映射
   b. `persist`  ：ŵ_d = 过去 N 个交易日 w*_的滚动均值（最宽容的方案 ——
                   不需要任何 regime 变量，只要最优权重有**持续性**就能赚到）
   若连 `persist` 都拿不到收益，说明 w*_d 逐日几乎是纯噪声，
   **那么没有任何条件化方案（if-else / 平滑门 / 更好的 regime 变量）能成功**，
   与样本量或参数化都无关。

专家取 T115_idxrel_s42（生产件，纯加性，从未见过门）并冻结，与 T128 同口径。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.nam_gate_model import NAMGateModel
from core.factors.regime_features import build_regime_matrix
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp.diag_frozen_expert_gate import precompute_group_sums
from scripts.exp.exp_nam_gate import _day_slices, _prepare_fold, _rank_ic_columns


def _rank01(x):
    o = np.argsort(np.argsort(x))
    return (o + 0.5) / len(x)


def daily_optimal_w(S, days, ret, ridge=1.0):
    """逐日岭回归最优族权重 w*_d，以及 oracle / 基线的当日 IC。

    目标取当日收益的**截面 rank**（与 rank IC 的口径一致），
    特征取族输出（逐日标准化，使 ridge 的惩罚在各族间可比）。
    """
    K = S.shape[1]
    W = np.full((len(days), K), np.nan)
    ic_or = np.full(len(days), np.nan)
    ic_base = np.full(len(days), np.nan)
    for i, (s, e) in enumerate(days):
        if e - s < 30:
            continue
        G = S[s:e].astype(np.float64)
        mu, sd = G.mean(0), G.std(0)
        sd[sd < 1e-12] = 1.0
        Z = (G - mu) / sd
        r = np.asarray(ret[s:e], dtype=np.float64)
        t = _rank01(r) - 0.5
        A = Z.T @ Z + ridge * len(Z) * np.eye(K)
        w = np.linalg.solve(A, Z.T @ t)
        W[i] = w
        ic_or[i] = _rank_ic_columns((Z @ w)[:, None], r)[0]
        ic_base[i] = _rank_ic_columns(Z.sum(1)[:, None], r)[0]
    return W, ic_or, ic_base


def _ic_with_w(S, days, ret, W):
    out = np.full(len(days), np.nan)
    for i, (s, e) in enumerate(days):
        if e - s < 30 or not np.all(np.isfinite(W[i])):
            continue
        G = S[s:e].astype(np.float64)
        mu, sd = G.mean(0), G.std(0)
        sd[sd < 1e-12] = 1.0
        Z = (G - mu) / sd
        out[i] = _rank_ic_columns((Z @ W[i])[:, None], np.asarray(ret[s:e], dtype=np.float64))[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/nam_gate/T115_idxrel_s42')
    ap.add_argument('--cache-dir', default='database/system_data/factors_cache_2026-08-18-idxrel')
    ap.add_argument('--stocks', type=int, default=6000)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--ridge', type=float, default=1.0)
    ap.add_argument('--persist-windows', default='5,20,60,120,250')
    ap.add_argument('--embargo', type=int, default=7,
                    help='persist 臂的禁运交易日数。w*_j 由 j 日的**前向 7 日**收益解出，'
                         '要到 j+7 才可知，而今天的目标又覆盖 i..i+6 —— 不禁运时窗口末端'
                         '与目标区间大面积重叠，等于偷看未来（T130 第一版实测因此虚增）。')
    ap.add_argument('--out', default='diagnose_output/T129_oracle_ceiling.json')
    a = ap.parse_args()

    m = NAMGateModel()
    m.load_model(os.path.join(a.model, 'nam_gate_factor_model.pkl'))
    feats = list(m.feature_names)
    K = len(m.group_names)
    print(f'冻结专家 {a.model}: {len(feats)} 列 / {K} 族')

    end_dt = datetime.strptime(a.end, '%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * a.years)).strftime('%Y-%m-%d')
    TrainingConfig.FUTURE_DAYS = 7
    tr = MLModelTrainer(db_path=DATABASE_PATH, cache_dir=a.cache_dir)
    mgr = BaostockDataManager()
    codes = mgr.get_stock_list_from_db()['code'].tolist()[:a.stocks]
    mgr.close()
    sd_ = tr.load_label_data(codes, start, a.end)
    ds = tr.prepare_dataset(sd_, train_start_date=start, train_end_date=a.end,
                            include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
                            n_jobs=4, target_features=None, use_factor_cache_only=True)
    regime = build_regime_matrix(DATABASE_PATH)
    fold = _prepare_fold(tr, ds, feats, 0.8, 1.0, regime, target='returns')
    del sd_, ds

    gid = np.asarray(m.group_ids)
    S_tr = precompute_group_sums(m, fold['X_train'], gid, K).cpu().numpy()
    S_va = precompute_group_sums(m, fold['X_val'], gid, K).cpu().numpy()
    tr_days, va_days = _day_slices(fold['d_train']), _day_slices(fold['d_val'])
    M_tr = np.stack([fold['M_train'][s] for s, _ in tr_days])
    M_va = np.stack([fold['M_val'][s] for s, _ in va_days])

    print('\n=== ① oracle 天花板（允许看当天未来收益）===')
    Wtr, or_tr, ba_tr = daily_optimal_w(S_tr, tr_days, fold['ret_train'], a.ridge)
    Wva, or_va, ba_va = daily_optimal_w(S_va, va_days, fold['ret_val'], a.ridge)
    print(f'  训练段 {np.isfinite(or_tr).sum()} 日: 基线 IC {np.nanmean(ba_tr):+.5f} '
          f'→ oracle {np.nanmean(or_tr):+.5f}  (×{np.nanmean(or_tr)/max(np.nanmean(ba_tr),1e-9):.2f})')
    print(f'  验证段 {np.isfinite(or_va).sum()} 日: 基线 IC {np.nanmean(ba_va):+.5f} '
          f'→ oracle {np.nanmean(or_va):+.5f}  (×{np.nanmean(or_va)/max(np.nanmean(ba_va),1e-9):.2f})')
    print('  ⇒ 天花板远高于基线 = 「逐日最优权重确实存在」，信息不是不存在。')

    ok_tr = np.all(np.isfinite(Wtr), axis=1)
    ok_va = np.all(np.isfinite(Wva), axis=1)

    print('\n=== ② w*_d 能被 regime 预测多少（训练段拟合，验证段折外 R²）===')
    X = np.column_stack([np.ones(ok_tr.sum()), M_tr[ok_tr]])
    Xv = np.column_stack([np.ones(ok_va.sum()), M_va[ok_va]])
    B = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Wtr[ok_tr])
    pred_va = Xv @ B
    r2 = []
    for k in range(K):
        y = Wva[ok_va, k]
        ss = ((y - y.mean()) ** 2).sum()
        r2.append(1 - ((y - pred_va[:, k]) ** 2).sum() / max(ss, 1e-12))
    print(f'{"族":12s} {"折外R²":>9s}')
    for k in np.argsort(r2)[::-1]:
        print(f'{m.group_names[k]:12s} {r2[k]:+9.4f}')
    print(f'  折外 R² 中位 {np.median(r2):+.4f}  均值 {np.mean(r2):+.4f}   [≤0 = 还不如用常数]')

    print('\n=== ③ 两个可实现方案（只用过去信息）在验证段的真实 IC ===')
    print(f'{"方案":22s} {"验证段IC":>10s} {"Δ vs 基线":>11s}')
    base = np.nanmean(ba_va)
    print(f'{"基线(等权 w≡1)":22s} {base:+10.5f} {0.0:+11.5f}')
    Wp = np.full_like(Wva, np.nan)
    Wp[ok_va] = pred_va
    ic_reg = _ic_with_w(S_va, va_days, fold['ret_val'], Wp)
    print(f'{"regime 线性映射":22s} {np.nanmean(ic_reg):+10.5f} {np.nanmean(ic_reg)-base:+11.5f}')
    res_persist = {}
    allW = np.vstack([Wtr, Wva])
    n_tr = len(Wtr)
    for N in [int(x) for x in a.persist_windows.split(',')]:
        Wr = np.full_like(Wva, np.nan)
        for i in range(len(Wva)):
            hi = n_tr + i - a.embargo          # 禁运：w*_j 要到 j+7 才可知
            lo = max(0, hi - N)
            hist = allW[lo:max(0, hi)]
            hist = hist[np.all(np.isfinite(hist), axis=1)]
            if len(hist) >= max(3, N // 4):
                Wr[i] = hist.mean(0)
        ic = _ic_with_w(S_va, va_days, fold['ret_val'], Wr)
        res_persist[N] = float(np.nanmean(ic))
        print(f'{f"过去{N}日滚动均值(禁运{a.embargo})":22s} {np.nanmean(ic):+10.5f} '
              f'{np.nanmean(ic)-base:+11.5f}')

    print('\n=== ④ w*_d 的持续性（相邻日自相关）===')
    Wv = Wva[ok_va]
    ac = [np.corrcoef(Wv[:-1, k], Wv[1:, k])[0, 1] for k in range(K)]
    print(f'  逐族一阶自相关: 中位 {np.median(ac):+.4f}  范围 [{min(ac):+.4f}, {max(ac):+.4f}]')
    print('  ⇒ 接近 0 意味着「今天的最优权重」对明天几乎没有信息，')
    print('     那么任何条件化方案（if-else / 平滑门 / 换 regime 变量）都无处着力。')

    out = {'model': a.model,
           'oracle': {'train_base': float(np.nanmean(ba_tr)), 'train_oracle': float(np.nanmean(or_tr)),
                      'val_base': float(base), 'val_oracle': float(np.nanmean(or_va))},
           'r2_oos': {m.group_names[k]: float(r2[k]) for k in range(K)},
           'realizable': {'regime_linear': float(np.nanmean(ic_reg)), 'persist': res_persist},
           'autocorr': {m.group_names[k]: float(ac[k]) for k in range(K)}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
