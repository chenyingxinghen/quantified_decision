#!/usr/bin/env python
"""T130：**头部可靠性的 regime 依赖** —— 用户「风险偏好」假说的直接检验。

用户的假说（与门控轴测过的东西**不是同一件事**）
------------------------------------------------
> 「熊市也有熊市的牛股，从特征到排序这个映射关系是类似的（至少在头部来看）。
>   关键在于：环境好可以大胆抓头部；环境差必须稳健 —— 头部也许仍然涨势很好，
>   但猜中的概率太小。不同市场环境下，模型的关注点必须不一样。」

这与 T116~T129 测的命题**正交**，必须分开：

  命题 A（门控轴测的）：**全截面排序函数**随 regime 变化。
      判据：全截面 rank IC。结论：2×2 四格全负、冻结专家后 Δ=−0.00025(2/4)、
      `corr(IC_k,s)` 训练→验证符号一致率 50%。**已封口。**

  命题 B（用户现在说的）：排序函数基本不变，但**头部的可靠性**随 regime 变化。
      判据：top-K 命中率 / 头部超额。**从来没测过。**

⚠ 为什么两者可以**同时成立**（这条是本脚本存在的理由）：
  rank IC 在 ~4700 只的全截面上算，我们**只买前 20 只**。把头部 20 只重排一遍，
  全截面 rank IC 几乎不动。所以我们那套晋级协议（4/4 且全截面 ΔIC ≥0.005）
  **在结构上看不见头部改动**。门控轴的所有否定结论对命题 A 有效，对命题 B 无效。

⚠ 但架构给了命题 B 一个硬约束（必须先说清，否则测了也没用）：
  策略每天按分数取前 K 只、等权满仓。所以「风险偏好」在这套架构里只有三种出口：
    (a) 降仓位 / 持币  → 择时风控规则，用户已禁止，且它是 β 不是 α；
    (b) 放宽 K（分散） → 组合构建规则；K 轴已扫过单峰在 20，**但 regime 条件 K 没测过**；
    (c) **换掉选的是哪 20 只** → 只有这条是 alpha，也只有这条在本脚本射程内。
  所以 ② 测的就是 (c)：在头部候选池内部重排，能不能靠 regime 条件化赚到钱。

三段测量
--------
① **前提检验**：头部可靠性到底随不随 regime 变？关键是**扣掉两个平凡解释**：
     - 全截面 IC 本身就在跌日更低（那不是新信息，是命题 A 的老结论）；
     - 跌日**截面离散度**σ 本身就小（超额自然小，但这不可修复 —— 你没法靠
       换股票把一个没有分化的市场变得有分化）。
   所以除了原始 topK 超额，还报 **σ 归一化超额** `excess/σ_d` 与 **命中率**。
   判读：若归一化超额与命中率的跌/涨比 ≈ IC 的跌/涨比 ⇒ 头部可靠性就是 IC，
   用户的直觉被 IC 完全解释，无新东西；若前者掉得**明显更多** ⇒ 命题 B 有内容。

② **头部内重排的 oracle 天花板 + 可实现方案**（全部只用过去信息，验证段评估）：
     production  按冻结分数取前 K（就是当前生产件本身）        ← 配对基线
     global      训练段拟合**一个**族权重（不看 regime）        ← 关键对照：重排本身有没有用
     ifelse      训练段按「好/坏环境」各拟合一个，验证段按当日标签选用 ← 用户要的那个 if-else
     ifelse_oracle 同上，但用**事后**市场涨跌标签              ← if-else 的天花板
     regime      ŵ_d = 训练段拟合的 30 维 m_d → w 线性映射
     persist_N   ŵ_d = 过去 N 日 w*_d 的滚动均值（不需要任何 regime 变量）
     oracle      w*_d 直接用当天未来收益解出                    ← 天花板，不可实现
   若 `global` 就没用 ⇒ 头部重排这条路本身是死的；
   若 `global` 有用但 `ifelse`/`regime` 不比它好 ⇒ 增益与 regime 无关；
   若连 `oracle` 都不比 production 高多少 ⇒ 头部内**根本没有可重排的信息**。

③ **头部 vs 全截面的对照**：同一套权重在 `--head-n 0`（全截面）下重跑 ①②，
   直接量出「换成头部尺子后，结论差多少」。这是本脚本相对 T129 的增量。

专家取 T115_idxrel_s{42,11,23,37}（生产件，纯加性，从未见过门）并冻结，与 T128/T129 同口径。
判据用**逐日配对 top-K 超额** + 4 专家配对 Δ + **跌日分层否决门** —— 注意这套判据与
「全截面 ΔIC ≥0.005」是两把不同的尺子，头部改动只能用前者判。

⚠ 第一版踩的坑（必须记住，它差点让我报一个假阳性）
   `persist` 臂拿「过去 N 日的 w*_d 滚动均值」当今天的权重，看似只用过去信息 ——
   **错**。`w*_j` 是用 j 日的**前向 7 日收益**解出来的，要到 j+7 才可知；
   而今天 i 的目标又覆盖 i..i+6。不加禁运时窗口末端 6 天与目标区间**大面积重叠**，
   等于直接偷看。第一版 `persist20` 因此报出 Δ=+0.0089 / t=+6.40 / hit 0.664 的「大赢」。
   现在 `--embargo 7` 把窗口截到 `i-7`，这条线必须重测。

⚠ 口径说明：本诊断**不过滤** unbuyable / ST（生产回测会过滤）。它问的是
  「信息在不在头部」，不是「能不能买到」；要回答后者得跑回测。
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
from scipy.stats import rankdata, ttest_rel

from config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.nam_gate_model import NAMGateModel
from core.factors.regime_features import build_regime_matrix
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp.diag_frozen_expert_gate import precompute_group_sums
from scripts.exp.exp_nam_gate import _day_slices, _prepare_fold


def _rank01(x):
    o = np.argsort(np.argsort(x))
    return (o + 0.5) / len(x)


def build_days(S, ret, days, head_n, min_rows=30):
    """把每个交易日打包成头部候选池：pool 索引、池内标准化 Z、目标 t、全日均值。

    pool = 按**冻结分数**（raw 族和的行和）取前 head_n；head_n<=0 表示全截面。
    Z 在池内标准化只为让 ridge 惩罚在各族间可比；因为按 `Z@w` 排序等价于按
    `G@(w/σ)` 排序，所以标准化不改变可达的排序集合，天花板不受影响。
    """
    out = []
    for s, e in days:
        n = e - s
        if n < min_rows:
            continue
        G_all = S[s:e].astype(np.float64)
        r_all = np.asarray(ret[s:e], dtype=np.float64)
        base = G_all.sum(1)
        if head_n and head_n < n:
            pool = np.argpartition(base, -head_n)[-head_n:]
        else:
            pool = np.arange(n)
        G = G_all[pool]
        mu, sd = G.mean(0), G.std(0)
        sd[sd < 1e-12] = 1.0
        out.append({
            'Z': (G - mu) / sd,
            'base': base[pool],                  # 池内的生产分数（原始尺度）
            'r': r_all[pool],
            't': _rank01(r_all[pool]) - 0.5,
            'mkt': float(r_all.mean()),          # 当日全截面等权前向收益 = 「环境」
            'sig': float(r_all.std()),           # 当日截面离散度
            'r_all': r_all,
            'base_all': base,
        })
    return out


def daily_optimal_w(pack, ridge=1.0):
    """逐日岭回归最优族权重 w*_d（池内），闭式解。"""
    K = pack[0]['Z'].shape[1]
    W = np.full((len(pack), K), np.nan)
    for i, d in enumerate(pack):
        Z = d['Z']
        A = Z.T @ Z + ridge * len(Z) * np.eye(K)
        W[i] = np.linalg.solve(A, Z.T @ d['t'])
    return W


def oracle_cv_scores(pack, ridge=1.0, seed=0):
    """当日**折外** oracle：池内随机二分，各用另一半拟合的 w 打分，拼回整池。

    朴素 oracle（同一批样本既拟合又打分，200 行拟 12 参）会把当天的噪声也吃进去，
    报出来的天花板是虚高的。折外版每只票的分都来自没见过它的拟合，是诚实上界。
    """
    rng = np.random.default_rng(seed)
    out = []
    for d in pack:
        Z, t = d['Z'], d['t']
        n, K = Z.shape
        half = rng.permutation(n) < n // 2
        sc = np.empty(n)
        for msk in (half, ~half):
            oth = ~msk
            if msk.sum() < K + 2 or oth.sum() < K + 2:
                sc[msk] = d['base'][msk]
                continue
            Zo, to = Z[oth], t[oth]
            w = np.linalg.solve(Zo.T @ Zo + ridge * len(Zo) * np.eye(K), Zo.T @ to)
            sc[msk] = Z[msk] @ w
        out.append(sc)
    return out


def pooled_w(pack, idx, ridge=1.0):
    """把若干天的池内样本**合并**成一个回归，拟合单一权重向量（global / if-else 用）。"""
    if len(idx) == 0:
        return None
    K = pack[0]['Z'].shape[1]
    A = np.zeros((K, K))
    b = np.zeros(K)
    n = 0
    for i in idx:
        Z, t = pack[i]['Z'], pack[i]['t']
        A += Z.T @ Z
        b += Z.T @ t
        n += len(Z)
    A += ridge * n * np.eye(K)
    return np.linalg.solve(A, b)


def eval_scheme(pack, W, k, scores=None):
    """给定逐日权重 W（nan 行 = 当天没有可用权重，回落到 production），逐日算三个量。

    也可直接传 scores（逐日打分数组列表），用于折外 oracle 这类不是「一个 w」的方案。
    """
    ex = np.full(len(pack), np.nan)
    hit = np.full(len(pack), np.nan)
    ic = np.full(len(pack), np.nan)
    for i, d in enumerate(pack):
        if scores is not None:
            sc = scores[i]
        elif W is None or not np.all(np.isfinite(W[i])):
            sc = d['base']
        else:
            sc = d['Z'] @ W[i]
        r = d['r']
        kk = min(k, len(r))
        top = np.argpartition(sc, -kk)[-kk:]
        ex[i] = r[top].mean() - d['r_all'].mean()
        hit[i] = float((r[top] > np.median(d['r_all'])).mean())
        if len(r) > 2:
            ic[i] = np.corrcoef(rankdata(sc), rankdata(r))[0, 1]
    return ex, hit, ic


def full_day_stats(pack, ks):
    """① 前提检验用的逐日量：全截面 rank IC + 各 K 的超额/命中率 + σ + 市场收益。"""
    n = len(pack)
    ic = np.full(n, np.nan)
    sig = np.array([d['sig'] for d in pack])
    mkt = np.array([d['mkt'] for d in pack])
    ex = {k: np.full(n, np.nan) for k in ks}
    hit = {k: np.full(n, np.nan) for k in ks}
    for i, d in enumerate(pack):
        b, r = d['base_all'], d['r_all']
        ic[i] = np.corrcoef(rankdata(b), rankdata(r))[0, 1]
        med = np.median(r)
        for k in ks:
            kk = min(k, len(r))
            top = np.argpartition(b, -kk)[-kk:]
            ex[k][i] = r[top].mean() - r.mean()
            hit[k][i] = float((r[top] > med).mean())
    return ic, ex, hit, sig, mkt


def _t(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float('nan')
    return float(ttest_rel(a[m], b[m]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(
        f'models/nam_gate/T115_idxrel_s{s}' for s in (42, 11, 23, 37)))
    ap.add_argument('--cache-dir', default='database/system_data/factors_cache_2026-08-18-idxrel')
    ap.add_argument('--cache-npz', default='diagnose_output/T130_groupsums.npz',
                    help='族输出缓存；存在则跳过 ~20min 取数，便于反复改判读口径')
    ap.add_argument('--stocks', type=int, default=6000)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--head-n', default='200,0', help='头部候选池大小，逗号分隔；0 = 全截面')
    ap.add_argument('--regime-col', default='trend_ma60_dev',
                    help='① 分层与 ③ if-else 用的**事前**环境标签列')
    ap.add_argument('--ridge', type=float, default=1.0)
    ap.add_argument('--embargo', type=int, default=7,
                    help='persist 臂的禁运交易日数。w*_j 由 j 日的**前向 7 日**收益解出，'
                         '要到 j+7 才可知；不禁运 = 直接偷看未来（第一版就踩了）')
    ap.add_argument('--persist-windows', default='20,60,250')
    ap.add_argument('--out', default='diagnose_output/T130_head_regime.json')
    a = ap.parse_args()

    dirs = [d.strip() for d in a.models.split(',') if d.strip()]
    seeds = [d.rsplit('_s', 1)[1] for d in dirs]
    m = NAMGateModel()
    m.load_model(os.path.join(dirs[0], 'nam_gate_factor_model.pkl'))
    feats = list(m.feature_names)
    gid = np.asarray(m.group_ids)
    K = len(m.group_names)
    ri = m.regime_cols.index(a.regime_col)
    print(f'冻结专家 {len(dirs)} 套: {len(feats)} 列 / {K} 族，环境标签 {a.regime_col}')

    if a.cache_npz and os.path.exists(a.cache_npz):
        z = np.load(a.cache_npz, allow_pickle=False)
        S_tr = [z[f'S_tr_{i}'] for i in range(len(dirs))]
        S_va = [z[f'S_va_{i}'] for i in range(len(dirs))]
        ret_tr, ret_va = z['ret_tr'], z['ret_va']
        tr_days = [(int(s), int(e)) for s, e in z['tr_days']]
        va_days = [(int(s), int(e)) for s, e in z['va_days']]
        M_tr, M_va = z['M_tr'], z['M_va']
        print(f'  已复用族输出缓存 {a.cache_npz}')
    else:
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
                                n_jobs=a.n_jobs, target_features=None, use_factor_cache_only=True)
        regime = build_regime_matrix(DATABASE_PATH)
        fold = _prepare_fold(tr, ds, feats, 0.8, 1.0, regime, target='returns')
        del sd_, ds
        S_tr, S_va = [], []
        for d in dirs:
            mi = NAMGateModel()
            mi.load_model(os.path.join(d, 'nam_gate_factor_model.pkl'))
            S_tr.append(precompute_group_sums(mi, fold['X_train'], gid, K).cpu().numpy())
            S_va.append(precompute_group_sums(mi, fold['X_val'], gid, K).cpu().numpy())
            del mi
            torch.cuda.empty_cache()
        fold['X_train'] = fold['X_val'] = None    # 专家已冻结算完，8GB 面板立刻释放
        ret_tr, ret_va = np.asarray(fold['ret_train']), np.asarray(fold['ret_val'])
        tr_days, va_days = _day_slices(fold['d_train']), _day_slices(fold['d_val'])
        M_tr = np.stack([fold['M_train'][s] for s, _ in tr_days])
        M_va = np.stack([fold['M_val'][s] for s, _ in va_days])
        if a.cache_npz:
            os.makedirs(os.path.dirname(os.path.abspath(a.cache_npz)), exist_ok=True)
            np.savez(a.cache_npz, ret_tr=ret_tr, ret_va=ret_va,
                     tr_days=np.array(tr_days), va_days=np.array(va_days),
                     M_tr=M_tr, M_va=M_va,
                     **{f'S_tr_{i}': v for i, v in enumerate(S_tr)},
                     **{f'S_va_{i}': v for i, v in enumerate(S_va)})
            print(f'  族输出已缓存到 {a.cache_npz}')
        del fold

    g_tr_all = M_tr[:, ri]
    g_va_all = M_va[:, ri]
    print(f'训练 {len(tr_days)} 日 / 验证 {len(va_days)} 日，top-K = {a.topk}，'
          f'persist 禁运 {a.embargo} 日')

    result = {'models': dirs, 'topk': a.topk, 'regime_col': a.regime_col,
              'embargo': a.embargo, 'variants': {}}
    keep_tr = [i for i, (s, e) in enumerate(tr_days) if e - s >= 30]
    keep_va = [i for i, (s, e) in enumerate(va_days) if e - s >= 30]
    gt, gv = g_tr_all[keep_tr], g_va_all[keep_va]
    Mt, Mv = M_tr[keep_tr], M_va[keep_va]

    for head_n in [int(x) for x in a.head_n.split(',')]:
        tag = f'head{head_n}' if head_n else 'full'
        print(f'\n{"=" * 78}\n### 候选池 = {"前 %d 只" % head_n if head_n else "全截面"}\n{"=" * 78}')
        per_seed, mkt_va = {}, None

        for mi, sdd in enumerate(seeds):
            p_tr = build_days(S_tr[mi], ret_tr, tr_days, head_n)
            p_va = build_days(S_va[mi], ret_va, va_days, head_n)
            mkt_tr = np.array([d['mkt'] for d in p_tr])
            mkt_va = np.array([d['mkt'] for d in p_va])

            # ---------- ① 前提检验（只做一次：s42 + 全截面口径）----------
            if head_n == 0 and mi == 0:
                _premise(p_va, a, mkt_va, gt, gv, result)

            Wtr = daily_optimal_w(p_tr, a.ridge)
            Wva = daily_optimal_w(p_va, a.ridge)
            ok_tr = np.all(np.isfinite(Wtr), axis=1)

            schemes = {'production': None}
            w_glob = pooled_w(p_tr, np.arange(len(p_tr)), a.ridge)
            schemes['global'] = np.repeat(w_glob[None], len(p_va), 0)
            thr_tr = np.median(gt)
            w_good = pooled_w(p_tr, np.where(gt > thr_tr)[0], a.ridge)
            w_bad = pooled_w(p_tr, np.where(gt <= thr_tr)[0], a.ridge)
            schemes['ifelse'] = np.where((gv > thr_tr)[:, None], w_good[None], w_bad[None])
            w_up = pooled_w(p_tr, np.where(mkt_tr > 0)[0], a.ridge)
            w_dn = pooled_w(p_tr, np.where(mkt_tr <= 0)[0], a.ridge)
            schemes['ifelse_oracle'] = np.where((mkt_va > 0)[:, None], w_up[None], w_dn[None])
            X = np.column_stack([np.ones(ok_tr.sum()), Mt[ok_tr]])
            B = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Wtr[ok_tr])
            schemes['regime'] = np.column_stack([np.ones(len(p_va)), Mv]) @ B

            allW, n_tr = np.vstack([Wtr, Wva]), len(Wtr)
            for N in [int(x) for x in a.persist_windows.split(',')]:
                Wr = np.full_like(Wva, np.nan)
                for i in range(len(Wva)):
                    hi = n_tr + i - a.embargo          # 禁运：w*_j 要到 j+7 才可知
                    h = allW[max(0, hi - N):max(0, hi)]
                    h = h[np.all(np.isfinite(h), axis=1)]
                    if len(h) >= max(3, N // 4):
                        Wr[i] = h.mean(0)
                schemes[f'persist{N}'] = Wr

            ex0, hit0, _ = eval_scheme(p_va, None, a.topk)
            rows = {}
            for nm, W in schemes.items():
                ex, hit, ic = eval_scheme(p_va, W, a.topk)
                rows[nm] = {'ex': ex, 'hit': float(np.nanmean(hit)),
                            'ic': float(np.nanmean(ic))}
            for nm, sc in (('oracle_cv', oracle_cv_scores(p_va, a.ridge)),
                           ('oracle_insample', None)):
                ex, hit, ic = eval_scheme(p_va, Wva if sc is None else None, a.topk, scores=sc)
                rows[nm] = {'ex': ex, 'hit': float(np.nanmean(hit)), 'ic': float(np.nanmean(ic))}
            rows['production']['ex'] = ex0
            per_seed[sdd] = {'rows': rows,
                             'cos_pre': float(w_good @ w_bad
                                              / (np.linalg.norm(w_good) * np.linalg.norm(w_bad))),
                             'cos_oracle': float(w_up @ w_dn
                                                 / (np.linalg.norm(w_up) * np.linalg.norm(w_dn))),
                             'ac': float(np.median([np.corrcoef(Wva[:-1, k], Wva[1:, k])[0, 1]
                                                    for k in range(K)]))}
            del p_tr, p_va

        result['variants'][tag] = _report(per_seed, seeds, mkt_va, a)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    return 0


def _report(per_seed, seeds, mkt_va, a):
    """跨 4 个冻结专家汇总：配对 Δ、正种子数、MDE，以及**跌日**分层否决门。"""
    names = list(per_seed[seeds[0]]['rows'])
    up, dn = mkt_va > 0, mkt_va <= 0
    print(f'\n=== ② 验证段实测（{len(mkt_va)} 日，配对基线 = production，'
          f'4 个冻结专家）===')
    print(f'{"方案":16s} {"ex@%d" % a.topk:>9s} {"Δ均值":>9s} {"正种子":>7s} {"MDE":>8s} '
          f'{"t(日配对)":>10s} {"Δ涨日":>9s} {"Δ跌日":>9s} {"跌日正":>7s} {"hit":>6s}')
    out = {}
    for nm in names:
        d = np.array([np.nanmean(per_seed[s]['rows'][nm]['ex']
                                 - per_seed[s]['rows']['production']['ex']) for s in seeds])
        ex = np.array([np.nanmean(per_seed[s]['rows'][nm]['ex']) for s in seeds])
        dd = np.array([np.nanmean((per_seed[s]['rows'][nm]['ex']
                                   - per_seed[s]['rows']['production']['ex'])[dn]) for s in seeds])
        du = np.array([np.nanmean((per_seed[s]['rows'][nm]['ex']
                                   - per_seed[s]['rows']['production']['ex'])[up]) for s in seeds])
        pooled = np.concatenate([per_seed[s]['rows'][nm]['ex']
                                 - per_seed[s]['rows']['production']['ex'] for s in seeds])
        tt = _t(pooled, np.zeros_like(pooled))
        mde = 2 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        hit = float(np.mean([per_seed[s]['rows'][nm]['hit'] for s in seeds]))
        out[nm] = {'excess': float(ex.mean()), 'delta': float(d.mean()),
                   'n_pos': int((d > 0).sum()), 'mde': float(mde), 't_daily': tt,
                   'delta_up': float(du.mean()), 'delta_down': float(dd.mean()),
                   'n_pos_down': int((dd > 0).sum()), 'hit': hit}
        mark = ('  ← 基线' if nm == 'production'
                else '  ← 不可实现' if nm.startswith('oracle') or nm == 'ifelse_oracle' else '')
        print(f'{nm:16s} {ex.mean():+9.4f} {d.mean():+9.4f} {int((d > 0).sum())}/{len(d):<5d} '
              f'{mde:8.4f} {tt:+10.2f} {du.mean():+9.4f} {dd.mean():+9.4f} '
              f'{int((dd > 0).sum())}/{len(d):<5d} {hit:6.3f}{mark}')
    cp = np.mean([per_seed[s]['cos_pre'] for s in seeds])
    co = np.mean([per_seed[s]['cos_oracle'] for s in seeds])
    ac = np.mean([per_seed[s]['ac'] for s in seeds])
    print(f'\n  w*_d 逐族一阶自相关中位（4 专家均值）：{ac:+.4f}')
    print(f'  if-else 两侧权重余弦：事前标签 {cp:+.4f} / 事后涨跌 {co:+.4f}'
          f'   [→ +1 = 两分支学到同一个东西，if-else 退化成 global]')
    out['_meta'] = {'cos_pre': float(cp), 'cos_oracle': float(co),
                    'autocorr_median': float(ac), 'n_days': int(len(mkt_va))}
    return out


def _premise(p_va, a, mkt_v, gt, gv, result):
    """① 前提检验：头部可靠性随 regime 变吗（扣掉 IC 与截面离散度两个平凡解释）。"""
    ks = sorted({10, a.topk, 50, 100})
    ic_v, ex_v, hit_v, sig_v, _ = full_day_stats(p_va, ks)
    print(f'\n=== ① 前提检验：头部可靠性随 regime 变吗（验证段 {len(p_va)} 日）===')
    print('  分层 A：事后市场涨跌（当日全截面等权 7 日收益的符号）')
    print(f'{"分层":8s} {"日数":>5s} {"截面σ":>8s} {"全截面IC":>9s} '
          + ' '.join(f'{"ex@%d" % k:>9s} {"ex/σ@%d" % k:>9s} {"hit@%d" % k:>8s}' for k in ks))
    rows = {}
    for nm, msk in (('涨日', mkt_v > 0), ('跌日', mkt_v <= 0)):
        cells = [f'{msk.sum():5d}', f'{sig_v[msk].mean():8.4f}', f'{np.nanmean(ic_v[msk]):+9.4f}']
        rows[nm] = {'n_days': int(msk.sum()), 'sigma': float(sig_v[msk].mean()),
                    'ic': float(np.nanmean(ic_v[msk])), 'k': {}}
        for k in ks:
            e_, h_ = np.nanmean(ex_v[k][msk]), np.nanmean(hit_v[k][msk])
            en = np.nanmean(ex_v[k][msk] / np.maximum(sig_v[msk], 1e-9))
            cells += [f'{e_:+9.4f}', f'{en:+9.4f}', f'{h_:8.3f}']
            rows[nm]['k'][k] = {'excess': float(e_), 'excess_norm': float(en), 'hit': float(h_)}
        print(f'{nm:8s} ' + ' '.join(cells))
    u, d = rows['涨日'], rows['跌日']
    print('\n  ★ 判读（跌日 / 涨日 之比；<1 = 跌日更脆弱，用户假说成立）：')
    print(f'     全截面 IC        {d["ic"] / max(u["ic"], 1e-9):7.3f}')
    for k in ks:
        rn = (d['k'][k]['excess_norm'] / u['k'][k]['excess_norm']
              if abs(u['k'][k]['excess_norm']) > 1e-12 else np.nan)
        rh = ((d['k'][k]['hit'] - 0.5) / (u['k'][k]['hit'] - 0.5)
              if abs(u['k'][k]['hit'] - 0.5) > 1e-12 else np.nan)
        print(f'     σ归一超额@{k:<3d}    {rn:7.3f}      命中率超出0.5的部分 {rh:7.3f}')
    result['premise'] = {'strata': rows,
                         'sigma_ratio': float(d['sigma'] / max(u['sigma'], 1e-12))}

    thr = np.median(gt)
    print(f'\n  分层 B：事前标签 {a.regime_col}（训练段中位数 {thr:.4f} 为界）')
    pre = {}
    for nm, msk in (('好环境', gv > thr), ('坏环境', gv <= thr)):
        if msk.sum() < 20:
            continue
        en = np.nanmean(ex_v[a.topk][msk] / np.maximum(sig_v[msk], 1e-9))
        pre[nm] = {'n_days': int(msk.sum()), 'ic': float(np.nanmean(ic_v[msk])),
                   'excess': float(np.nanmean(ex_v[a.topk][msk])), 'excess_norm': float(en),
                   'hit': float(np.nanmean(hit_v[a.topk][msk])), 'mkt': float(mkt_v[msk].mean())}
        print(f'    {nm:8s} 日数 {msk.sum():4d}  市场 {mkt_v[msk].mean():+.4f}  '
              f'IC {np.nanmean(ic_v[msk]):+.4f}  ex@{a.topk} '
              f'{np.nanmean(ex_v[a.topk][msk]):+.4f}  ex/σ {en:+.4f}  '
              f'hit {np.nanmean(hit_v[a.topk][msk]):.3f}')
    result['premise_exante'] = pre

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
