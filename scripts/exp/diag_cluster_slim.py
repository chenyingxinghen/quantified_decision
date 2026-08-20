#!/usr/bin/env python
"""T121 零训练测量：224 列里的冗余是**真冗余**，还是在给加性通道做**噪声平均**？

背景（T117）：手工分族信息量为零，但数据里有强块结构 —— 数据驱动簇内聚 0.35~0.50、
相关矩阵 top12 主成分吃掉 75.2% 方差 ⇒ 224 列的预测行为只有约 3~12 个自由度。
台账当时留了一条建议：瘦身不该「删死列」（T114 已判负），而该**按数据簇选代表列**。

但这条建议的先验是负的，必须先测：加性 NAM 是 `score = Σ fᵢ(xᵢ)`，簇内相关 0.35~0.50
的成员各自是同一信号的噪声估计，**求和本身就在做噪声平均**。删掉冗余列保住了信号
方向，却丢了 √n 降噪 —— 这正是 T114 删列单调劣化的机制解释。

本脚本不训练，只用 T115_s42 已训好的形状函数做前向，回答四个口径在**无偏 holdout**
上的 rank IC（判定用的那一段，与 `--select-holdout 0.4` 同切法：验证折按时间后 40%）：

  ① full     : Σ 全部 224 列贡献（= 现基线的打分，参照系）
  ② drop     : 只留 K 个代表列，直接求和（"删掉其余列"的朴素做法）
  ③ rep_lsq  : 只留 K 个代表列，但系数在**训练段**最小二乘拟合 full 打分
               —— 这是「重训后代表列被允许重新定标」能达到的上界的代理
  ④ cmean_lsq: 每簇换成**簇内均值**，K 个系数同样训练段拟合
               —— 保住噪声平均、只压参数量的另一条路（Option B）

判读
----
* ② ≈ ① ⇒ 冗余是真冗余，瘦身值得砸训练预算
* ② ≪ ① 而 ④ ≈ ① ⇒ 冗余在做噪声平均，**该做的是簇聚合而不是删列**
* ③ ≪ ① ⇒ 连"允许重新定标"都救不回来，代表列这条路直接关掉，不用重训

⚠ 边界：③/④ 固定了形状函数（用 T115 训出来的那套），真重训能重新拟合形状。
  所以本测量对「不该做」是强证据，对「该做」只是必要条件。

用法：
  python scripts/exp/diag_cluster_slim.py --k-list 12,24,48,96,160
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

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
from scripts.exp.diag_factor_clustering import daily_ic_matrix
from scripts.exp.exp_nam_gate import _day_slices, _prepare_fold, _rank_ic_columns


def _contrib_day(model, Xrow, mean, std, dev):
    """单日的逐列专家贡献 [B, N]，已按日截面去均值（ListNet 逐日平移不变）。"""
    xt = torch.as_tensor(np.asarray(Xrow, dtype=np.float32), device=dev)
    if mean is not None:
        xt = (xt - torch.as_tensor(mean, device=dev)) / torch.as_tensor(std, device=dev)
    with torch.no_grad():
        c = model.net.experts(xt).cpu().numpy().astype(np.float64)
    return c - c.mean(axis=0, keepdims=True)


def _pick_reps(C: np.ndarray, labels: np.ndarray, ok: np.ndarray,
               ic_abs: np.ndarray, how: str) -> np.ndarray:
    """每簇挑一个代表列，返回全局列号数组。

    medoid : 与簇内其他成员平均相关最高（纯结构，不看标签）
    icmax  : 训练段 |贡献 IC| 最大（只用训练段标签，不碰验证段）
    """
    idx_ok = np.flatnonzero(ok)
    pos = {g: i for i, g in enumerate(idx_ok)}
    reps = []
    for c in np.unique(labels[labels >= 0]):
        members = np.flatnonzero((labels == c) & ok)
        if len(members) == 0:
            continue
        if len(members) == 1 or how == 'icmax':
            reps.append(members[int(np.argmax(ic_abs[members]))])
            continue
        sub = C[np.ix_([pos[i] for i in members], [pos[i] for i in members])]
        np.fill_diagonal(sub, np.nan)
        reps.append(members[int(np.nanargmax(np.nanmean(sub, axis=1)))])
    return np.asarray(sorted(set(reps)), dtype=int)


def _agg_matrix(labels: np.ndarray, n_feat: int) -> np.ndarray:
    """簇均值算子 [N, K]：contrib @ A = 各簇内均值。"""
    cs = np.unique(labels[labels >= 0])
    A = np.zeros((n_feat, len(cs)), dtype=np.float64)
    for j, c in enumerate(cs):
        m = np.flatnonzero(labels == c)
        A[m, j] = 1.0 / len(m)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/nam_gate/T115_idxrel_s42')
    ap.add_argument('--cache-dir', default='database/system_data/factors_cache_2026-08-18-idxrel')
    ap.add_argument('--stocks', type=int, default=6000)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--select-holdout', type=float, default=0.4)
    ap.add_argument('--train-days', type=int, default=900,
                    help='聚类用的训练段抽样日数（与 T117 同口径）')
    ap.add_argument('--fit-days', type=int, default=600,
                    help='最小二乘拟合抽样的训练日数（0=全量，流式累加不吃内存但耗时）')
    ap.add_argument('--k-list', default='12,24,48,96,160')
    ap.add_argument('--out', default='diagnose_output/T121_cluster_slim.json')
    a = ap.parse_args()

    from sklearn.cluster import AgglomerativeClustering

    model = NAMGateModel()
    model.load_model(os.path.join(a.model, 'nam_gate_factor_model.pkl'))
    feats = list(model.feature_names)
    n_feat = len(feats)
    print(f'模型 {a.model}: {n_feat} 列')
    dev = torch.device(model.device)
    mean = None if model.input_mean is None else np.asarray(model.input_mean, dtype=np.float32)
    std = None if model.input_std is None else np.asarray(model.input_std, dtype=np.float32)
    model.net.eval()

    end_dt = datetime.strptime(a.end, '%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * a.years)).strftime('%Y-%m-%d')
    TrainingConfig.FUTURE_DAYS = 7
    trainer = MLModelTrainer(db_path=DATABASE_PATH, cache_dir=a.cache_dir)
    mgr = BaostockDataManager()
    codes = mgr.get_stock_list_from_db()['code'].tolist()[:a.stocks]
    mgr.close()
    stocks_data = trainer.load_label_data(codes, start, a.end)
    dataset = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=a.end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=4, target_features=None, use_factor_cache_only=True)
    regime = build_regime_matrix(DATABASE_PATH)
    fold = _prepare_fold(trainer, dataset, feats, 0.8, 1.0, regime, target='returns')
    del stocks_data, dataset

    # ── 聚类：只用训练段（拿验证段聚类＝把验证信息带进结构选择）──────────────
    print(f'\n=== 训练段逐日贡献 IC（抽样 {a.train_days} 日）→ 相关矩阵 ===')
    ic = daily_ic_matrix(model, fold['X_train'], fold['ret_train'], fold['d_train'],
                         max_days=a.train_days)
    ic = ic[~np.all(np.isnan(ic), axis=1)]
    ok = ~np.all(np.isnan(ic), axis=0)
    C = pd.DataFrame(ic[:, ok]).corr().to_numpy()
    ic_abs = np.nan_to_num(np.abs(np.nanmean(ic, axis=0)), nan=0.0)
    print(f'  IC 矩阵 {ic.shape}，参与聚类 {ok.sum()} 列')
    D = 1.0 - np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(D, 0.0)

    # ── holdout 切法与 --select-holdout 0.4 一致：验证折按时间后 40% ──────────
    va_days = _day_slices(fold['d_val'])
    n_sel = max(1, int(round(len(va_days) * (1.0 - a.select_holdout))))
    ho_days = va_days[n_sel:]
    print(f'  验证折 {len(va_days)} 日 → 选型 {n_sel} 日 / **报告 holdout {len(ho_days)} 日**')

    tr_days = _day_slices(fold['d_train'])
    if a.fit_days and len(tr_days) > a.fit_days:
        keep = np.unique(np.linspace(0, len(tr_days) - 1, a.fit_days).round().astype(int))
        tr_days = [tr_days[i] for i in keep]
    print(f'  最小二乘拟合用训练日 {len(tr_days)} 个')

    Ks = [int(x) for x in a.k_list.split(',') if x.strip()]
    plans = {}   # name -> (P 投影算子 [N, m], 说明)
    labels_by_k = {}
    for K in Ks:
        lab = np.full(n_feat, -1)
        lab[ok] = AgglomerativeClustering(n_clusters=min(K, int(ok.sum())),
                                          metric='precomputed',
                                          linkage='average').fit_predict(D)
        # 未参与聚类的列（IC 全 NaN）单独成簇，代表列口径下它们本就该被丢掉
        labels_by_k[K] = lab
        for how in ('medoid', 'icmax'):
            reps = _pick_reps(C, lab, ok, ic_abs, how)
            P = np.zeros((n_feat, len(reps)))
            P[reps, np.arange(len(reps))] = 1.0
            plans[f'K{K}_{how}'] = (P, len(reps))
        A = _agg_matrix(lab, n_feat)
        plans[f'K{K}_cmean'] = (A, A.shape[1])

    # ── 流式：训练段累加正规方程；holdout 段算四个口径的逐日 rank IC ──────────
    names = list(plans)
    G = {k: np.zeros((plans[k][1], plans[k][1])) for k in names}
    b = {k: np.zeros(plans[k][1]) for k in names}
    print('\n=== 训练段累加正规方程（拟合 full 打分）===')
    for i, (s, e) in enumerate(tr_days):
        if e - s < 20:
            continue
        c = _contrib_day(model, fold['X_train'][s:e], mean, std, dev)
        y = c.sum(axis=1)
        for k in names:
            Z = c @ plans[k][0]
            G[k] += Z.T @ Z
            b[k] += Z.T @ y
        if (i + 1) % 150 == 0:
            print(f'  {i+1}/{len(tr_days)}')
    coef = {}
    for k in names:
        m = G[k].shape[0]
        coef[k] = np.linalg.solve(G[k] + 1e-8 * np.trace(G[k]) / m * np.eye(m), b[k])

    print('\n=== holdout 段评估 ===')
    ics = {'full': []}
    for k in names:
        ics[f'{k}|drop'] = []
        ics[f'{k}|lsq'] = []
    for s, e in ho_days:
        if e - s < 20:
            continue
        c = _contrib_day(model, fold['X_val'][s:e], mean, std, dev)
        y = np.asarray(fold['ret_val'][s:e], dtype=np.float64)
        ics['full'].append(_rank_ic_columns(c.sum(axis=1)[:, None], y)[0])
        for k in names:
            Z = c @ plans[k][0]
            ics[f'{k}|drop'].append(_rank_ic_columns(Z.sum(axis=1)[:, None], y)[0])
            ics[f'{k}|lsq'].append(_rank_ic_columns((Z @ coef[k])[:, None], y)[0])

    full = float(np.nanmean(ics['full']))
    print(f'\nholdout rank IC（{len(ics["full"])} 日）  基线 full(224 列) = {full:+.5f}\n')
    print(f'{"口径":18s} {"列数":>5s} {"drop 直接求和":>14s} {"占比":>7s} '
          f'{"lsq 重定标":>12s} {"占比":>7s}')
    res = {'full': full, 'n_holdout_days': len(ics['full']), 'arms': {}}
    for K in Ks:
        for how in ('medoid', 'icmax', 'cmean'):
            k = f'K{K}_{how}'
            m = plans[k][1]
            d = float(np.nanmean(ics[f'{k}|drop']))
            l = float(np.nanmean(ics[f'{k}|lsq']))
            print(f'{k:18s} {m:5d} {d:+14.5f} {d/full*100:6.1f}% {l:+12.5f} {l/full*100:6.1f}%')
            res['arms'][k] = {'n_cols': m, 'drop_ic': d, 'lsq_ic': l,
                              'drop_ratio': d / full, 'lsq_ratio': l / full}
        print()
    for K in Ks:
        lab = labels_by_k[K]
        res.setdefault('cluster_sizes', {})[str(K)] = \
            {int(c): int((lab == c).sum()) for c in np.unique(lab[lab >= 0])}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'已保存: {a.out}')
    print('\n判读：drop≈100% ⇒ 真冗余，值得砸训练预算；'
          'drop≪100% 而 cmean 的 lsq≈100% ⇒ 冗余在做噪声平均，该聚合而不是删列；'
          'lsq 也≪100% ⇒ 连重新定标都救不回来，这条路直接关。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
