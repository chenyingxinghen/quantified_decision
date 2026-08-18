"""因子作用单元检验：手工分族 vs 数据驱动聚类，哪个才是门控该作用的单位。

要回答的问题
------------
门控只能看到族内求和 ``S_k = Σ_{i∈k} f_i(x_i)``，所以它能否起作用，取决于
**族内成员的预测行为是否同向**。若族内符号混杂，S_k 里已经互相抵消，门控
拿到的是被冲淡的合量（见台账 T116 节「簇内梯度抵消」）。

本脚本不训练，只做前向：
  1. 用已训练存档算出逐日 × 逐因子的**贡献 IC 矩阵** [n_days, n_factors]
  2. 度量「作用单元的内聚性」= 单元内成员日度 IC 序列的平均两两相关
  3. 三方对比：手工 12 族 / 数据驱动 12 簇 / 随机 12 分组（零假设基线）

判读
----
* 手工族 ≈ 随机分组  ⇒ 手工分族不携带信息（但不代表存在更好的分法）
* 数据簇 ≫ 手工族且 ≫ 随机 ⇒ **存在更好的作用单元**，软指派矩阵值得一跑
* 数据簇 ≈ 随机 ⇒ 因子的预测行为**本来就不成簇**，「作用单元错了」当场证伪，
  不必再花训练预算在重构分族上

另报告相关矩阵的特征值谱：若前几个主成分吃掉大部分方差，说明存在低秩块结构；
若谱平坦，说明没有簇可言。

用法:
  python scripts/exp/diag_factor_clustering.py --model models/nam_gate/T115_idxrel_s42
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
from config.factor_groups import build_group_index
from core.data.baostock_main import BaostockDataManager
from core.factors.nam_gate_model import NAMGateModel
from core.factors.regime_features import build_regime_matrix
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp.exp_nam_gate import _day_slices, _prepare_fold, _rank_ic_columns


def daily_ic_matrix(model, X, returns, dates, max_days=0):
    """[n_days, n_factors]：逐日、逐因子的贡献 rank IC。"""
    dev = torch.device(model.device)
    mean = None if model.input_mean is None else np.asarray(model.input_mean, dtype=np.float32)
    std = None if model.input_std is None else np.asarray(model.input_std, dtype=np.float32)
    slices = _day_slices(dates)
    if max_days and len(slices) > max_days:
        keep = np.unique(np.linspace(0, len(slices) - 1, max_days).round().astype(int))
        slices = [slices[i] for i in keep]
    out = np.full((len(slices), X.shape[1]), np.nan, dtype=np.float32)
    model.net.eval()
    with torch.no_grad():
        for di, (s, e) in enumerate(slices):
            if e - s < 20:
                continue
            xt = torch.as_tensor(np.asarray(X[s:e], dtype=np.float32), device=dev)
            if mean is not None:
                xt = (xt - torch.as_tensor(mean, device=dev)) / torch.as_tensor(std, device=dev)
            contrib = model.net.experts(xt).cpu().numpy().astype(np.float64)
            out[di] = _rank_ic_columns(contrib, np.asarray(returns[s:e], dtype=np.float64))
    return out


def cohesion(ic_mat: np.ndarray, labels: np.ndarray) -> tuple:
    """单元内平均两两相关（按单元规模加权）+ 各单元明细。"""
    ok = ~np.all(np.isnan(ic_mat), axis=0)
    C = pd.DataFrame(ic_mat[:, ok]).corr().to_numpy()
    lab = labels[ok]
    per, tot_w, tot_v = {}, 0.0, 0.0
    for g in np.unique(lab):
        idx = np.flatnonzero(lab == g)
        if len(idx) < 2:
            per[int(g)] = (len(idx), np.nan)
            continue
        sub = C[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        v = float(np.nanmean(sub[iu]))
        per[int(g)] = (len(idx), v)
        w = len(idx) * (len(idx) - 1) / 2
        tot_w += w
        tot_v += v * w
    return (tot_v / tot_w if tot_w else np.nan), per, C


def _merge_small(lab: np.ndarray, C: np.ndarray, ok: np.ndarray, min_size: int) -> np.ndarray:
    """把规模 < min_size 的簇并入相关性最近的大簇。

    单例/超小簇在门控里会复刻 status 病灶：贡献极小 ⇒ 门控权重不受数据约束 ⇒
    在 `w·f` 的平坦方向上漂移，再经 Σw=K 的零和约束饿死其他簇。
    """
    idx_ok = np.flatnonzero(ok)
    pos = {g: i for i, g in enumerate(idx_ok)}       # 全局列号 → 相关矩阵下标
    while True:
        sizes = pd.Series(lab[ok]).value_counts()
        small = [c for c, n in sizes.items() if n < min_size]
        big = [c for c, n in sizes.items() if n >= min_size]
        if not small or not big:
            break
        c = small[0]
        members = [pos[i] for i in np.flatnonzero((lab == c) & ok)]
        best, best_v = None, -np.inf
        for b in big:
            others = [pos[i] for i in np.flatnonzero((lab == b) & ok)]
            v = float(np.nanmean(C[np.ix_(members, others)]))
            if v > best_v:
                best, best_v = b, v
        lab[lab == c] = best
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/nam_gate/T115_idxrel_s42')
    ap.add_argument('--cache-dir', default='database/system_data/factors_cache_2026-08-18-idxrel')
    ap.add_argument('--stocks', type=int, default=6000)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--train-days', type=int, default=900, help='训练侧抽样日数（0=全量）')
    ap.add_argument('--n-random', type=int, default=200, help='随机分组零假设重复次数')
    ap.add_argument('--out', default='diagnose_output/T117_factor_clustering.json')
    ap.add_argument('--emit-group-map', default=None,
                    help='把**训练段**导出的簇标签写成 {feature: cluster} JSON，'
                         '供 exp_nam_gate.py --group-map-file 使用。'
                         '只用训练段：拿验证段聚类等于把验证信息带进架构选择。')
    ap.add_argument('--min-cluster-size', type=int, default=5,
                    help='小于该规模的簇合并进最近邻簇（按平均相关）。'
                         '单例簇会复刻 status 病灶：贡献极小 ⇒ 门控权重不受约束 ⇒ '
                         '在平坦方向漂移并经零和约束饿死其他簇。')
    a = ap.parse_args()

    from sklearn.cluster import AgglomerativeClustering

    model = NAMGateModel()
    model.load_model(os.path.join(a.model, 'nam_gate_factor_model.pkl'))
    feats = list(model.feature_names)
    group_names = list(model.group_names)
    print(f'模型 {a.model}: {len(feats)} 因子 / {len(group_names)} 族')

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

    _, group_ids = build_group_index(feats)
    hand = np.asarray(group_ids)

    res = {}
    for tag, (X, ret, d, md) in {
        'val': (fold['X_val'], fold['ret_val'], fold['d_val'], 0),
        'train': (fold['X_train'], fold['ret_train'], fold['d_train'], a.train_days),
    }.items():
        print(f'\n=== {tag} 段：算逐日贡献 IC ===')
        ic = daily_ic_matrix(model, X, ret, d, max_days=md)
        ic = ic[~np.all(np.isnan(ic), axis=1)]
        print(f'  IC 矩阵 {ic.shape}')

        h_val, h_per, C = cohesion(ic, hand)
        K = len(np.unique(hand))

        # 数据驱动聚类：以 1-相关 为距离做层次聚类，簇数与手工族相同
        D = 1.0 - np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(D, 0.0)
        cl = AgglomerativeClustering(n_clusters=K, metric='precomputed',
                                     linkage='average').fit_predict(D)
        ok = ~np.all(np.isnan(ic), axis=0)
        data_lab = np.full(len(hand), -1)
        data_lab[ok] = cl
        d_val_, d_per, _ = cohesion(ic, data_lab)

        # 零假设：保持手工族的规模分布，随机重排成员
        rng = np.random.default_rng(0)
        rnd = []
        for _ in range(a.n_random):
            perm = rng.permutation(len(hand))
            rnd.append(cohesion(ic, hand[perm])[0])
        rnd = np.asarray(rnd)

        # 特征值谱
        Cc = np.nan_to_num(C, nan=0.0)
        ev = np.sort(np.linalg.eigvalsh(Cc))[::-1]
        ev = ev / ev.sum()

        print(f'  内聚度（族内平均两两相关）:')
        print(f'    手工 {K} 族      : {h_val:+.4f}')
        print(f'    数据驱动 {K} 簇  : {d_val_:+.4f}')
        print(f'    随机分组均值     : {rnd.mean():+.4f}  (σ {rnd.std():.4f}, n={a.n_random})')
        z_hand = (h_val - rnd.mean()) / max(rnd.std(), 1e-9)
        z_data = (d_val_ - rnd.mean()) / max(rnd.std(), 1e-9)
        print(f'    手工 z = {z_hand:+.1f}   数据 z = {z_data:+.1f}')
        print(f'  特征值谱: top1={ev[0]:.1%} top3={ev[:3].sum():.1%} top12={ev[:12].sum():.1%}')

        res[tag] = {
            'n_days': int(ic.shape[0]), 'n_factors': int(ic.shape[1]),
            'hand_cohesion': float(h_val), 'data_cohesion': float(d_val_),
            'random_mean': float(rnd.mean()), 'random_std': float(rnd.std()),
            'z_hand': float(z_hand), 'z_data': float(z_data),
            'eig_top1': float(ev[0]), 'eig_top3': float(ev[:3].sum()),
            'eig_top12': float(ev[:12].sum()),
            'hand_per_group': {group_names[k]: [int(n), None if np.isnan(v) else float(v)]
                               for k, (n, v) in h_per.items() if k < len(group_names)},
            'data_cluster_sizes': {int(k): int(n) for k, (n, v) in d_per.items()},
        }
        if tag == 'val':
            res['val_data_cluster_members'] = {
                int(c): [feats[i] for i in np.flatnonzero(data_lab == c)][:40]
                for c in np.unique(cl)
            }
        if tag == 'train' and a.emit_group_map:
            lab = _merge_small(data_lab.copy(), C, ok, a.min_cluster_size)
            # 未参与聚类的列（全 NaN IC）归入最大簇，避免它们各自成为单例
            biggest = int(pd.Series(lab[lab >= 0]).value_counts().idxmax())
            lab[lab < 0] = biggest
            uniq = {c: i for i, c in enumerate(sorted(set(lab.tolist())))}
            gmap = {feats[i]: int(uniq[lab[i]]) for i in range(len(feats))}
            os.makedirs(os.path.dirname(os.path.abspath(a.emit_group_map)), exist_ok=True)
            with open(a.emit_group_map, 'w', encoding='utf-8') as f:
                json.dump(gmap, f, ensure_ascii=False, indent=1)
            sizes = pd.Series(list(gmap.values())).value_counts().sort_index()
            print(f'  已写簇映射 {a.emit_group_map}：{len(uniq)} 簇，规模 {sizes.to_dict()}')
            res['emitted_group_map'] = {'path': a.emit_group_map,
                                        'n_clusters': len(uniq),
                                        'sizes': {int(k): int(v) for k, v in sizes.items()}}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
