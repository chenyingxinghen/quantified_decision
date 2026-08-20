#!/usr/bin/env python
"""T128：**冻结专家、只训门** —— 用生产级"好模型"重做门控轴判定（批量实现）。

为什么必须做这个（用户提出的方法论质疑）
------------------------------------------
封轴证据分两类：
  ① T116/T118/T119/T125 vs T115 的**端到端配对**（同面板同超参同种子，只差门开关）
     —— 不依赖任何单模型内部读数，结论「联合训练下加门就是输」成立。
  ② T127 的「corr(IC_k, s) 训练→验证符号一致率 50%」—— **用 T116 的专家算的**，
     而 T116 正是被门带坏的模型。拿它下「条件结构在数据里不存在」这种强结论**站不住**。

本脚本消除该混淆：专家取 **T115_idxrel_s{42,11,23,37}（生产件，纯加性，从未见过门）**
并**冻结**，只训门。于是「门开 vs 门关」是**零其他差异**的配对 —— 专家逐位相同，
唯一变量是那 K 个门控参数。若条件结构存在，这是它最好的机会。

（用户原话是「先训门再训专家」。顺序须反过来：初始专家随机 ⇒ 族输出是噪声 ⇒
 门在噪声上学不到东西。正确形式是「先训加性 → 冻结专家 → 单独训门」，
 第一阶段产物就是已有的 T115 存档。）

两个实现要点（第一版踩的坑，记下来）
------------------------------------
1. **必须批量**。第一版逐日发 kernel（2727 日 × 每日几千行 × 12 列），
   实测 GPU 利用率仅 10% —— 纯 launch-bound，正是台账 chunk-days 那条教训。
   现在把逐日切片预打包成 `[C, L, K]` 块（块内补齐 + mask），
   一个 epoch 只需几十次 kernel。族输出只有 K 列，整个训练段打包也只有几百 MB。
2. **种子必须来自不同的冻结专家**。专家冻结后，全批梯度下降 + 零初始化是
   **确定性**的 —— 同一套专家跑四个"种子"会得到四个一模一样的数，
   报「4/4」是假的。所以种子维度取自 T115 的四个存档。

臂（每个冻结专家各跑一遍）
--------------------------
  additive   门关，w≡1                        —— 基线，就是该 T115 存档本身
  static     每族一个**自由静态权重**（不看 regime） —— 关键对照组
  scalar     w = K·softmax(a·s)，s=macro_m1m2_gap  —— T116/T118 的门
  × {手工 12 族, 数据驱动 6 簇} × {不归一化, 逐日截面归一化}

`static` 是要害：若增益主要来自它，门的价值就**不是 regime 条件化**，
而只是「重新分配族权重」这件与行情无关的事。

判定：无偏 holdout（验证折按时间后 40%，与 --select-holdout 0.4 同切法），
早停只看选型段（前 60%）。additive 在**完全相同的专家**上算，故为精确配对。
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
from scripts.exp.exp_nam_gate import _day_slices, _prepare_fold, _rank_ic_columns


def precompute_group_sums(model, X, group_ids, n_groups, batch=1 << 18):
    """冻结专家 ⇒ 族输出固定，一次性算完 [N, K]。"""
    dev = torch.device(model.device)
    mean = None if model.input_mean is None else np.asarray(model.input_mean, dtype=np.float32)
    std = None if model.input_std is None else np.asarray(model.input_std, dtype=np.float32)
    oh = torch.zeros(X.shape[1], n_groups, device=dev)
    oh[torch.arange(X.shape[1]), torch.as_tensor(np.asarray(group_ids), device=dev)] = 1.0
    out = torch.empty((len(X), n_groups), dtype=torch.float32, device=dev)
    model.net.eval()
    with torch.no_grad():
        for s in range(0, len(X), batch):
            e = min(s + batch, len(X))
            xt = torch.as_tensor(np.asarray(X[s:e], dtype=np.float32), device=dev)
            if mean is not None:
                xt = (xt - torch.as_tensor(mean, device=dev)) / torch.as_tensor(std, device=dev)
            out[s:e] = model.net.experts(xt) @ oh
    return out


def pack_days(S, days, y=None, chunk=512, norm=False):
    """逐日切片 → [C, L, K] 块列表（块内补齐 + mask）。norm=True 时按有效行做逐日标准化。"""
    dev = S.device
    K = S.shape[1]
    packs = []
    for c0 in range(0, len(days), chunk):
        blk = days[c0:c0 + chunk]
        C, L = len(blk), max(e - s for s, e in blk)
        G = torch.zeros(C, L, K, device=dev)
        M = torch.zeros(C, L, dtype=torch.bool, device=dev)
        Y = torch.zeros(C, L, device=dev) if y is not None else None
        for i, (s, e) in enumerate(blk):
            G[i, :e - s] = S[s:e]
            M[i, :e - s] = True
            if y is not None:
                Y[i, :e - s] = y[s:e]
        if norm:
            cnt = M.sum(1, keepdim=True).clamp_min(1).float()           # [C,1]
            mv = M.unsqueeze(-1).float()
            mu = (G * mv).sum(1) / cnt                                   # [C,K]
            var = (((G - mu.unsqueeze(1)) ** 2) * mv).sum(1) / cnt
            G = (G - mu.unsqueeze(1)) / (var.sqrt() + 1e-6).unsqueeze(1)
            G = G * mv
        packs.append((G, Y, M, c0, C))
    return packs


def _scores(packs, w_all, K):
    """批量算分：bmm([C,L,K],[C,K,1]) → [C,L]，不物化 C*L*K 的中间积。"""
    out = []
    for G, _, M, c0, C in packs:
        w = w_all[c0:c0 + C] if w_all.dim() == 2 else w_all.unsqueeze(0).expand(C, K)
        out.append((torch.bmm(G, w.unsqueeze(-1)).squeeze(-1), M, c0, C))
    return out


def _loss(packs, w_all, K, y_scale):
    tot = 0.0
    for (G, Y, M, c0, C) in packs:
        w = w_all[c0:c0 + C] if w_all.dim() == 2 else w_all.unsqueeze(0).expand(C, K)
        sc = torch.bmm(G, w.unsqueeze(-1)).squeeze(-1)
        neg = torch.finfo(sc.dtype).min
        logp = torch.log_softmax(sc.masked_fill(~M, neg), 1)
        tgt = torch.softmax((Y * y_scale).masked_fill(~M, neg), 1)
        tot = tot + (-(tgt * logp.masked_fill(~M, 0.0)).sum(1)).sum()
    return tot


def _daily_ic(packs, w_all, K, ret, days):
    ics = np.full(len(days), np.nan)
    for sc, M, c0, C in _scores(packs, w_all, K):
        sc = sc.detach().cpu().numpy()
        Mn = M.cpu().numpy()
        for i in range(C):
            s, e = days[c0 + i]
            v = sc[i][Mn[i]].astype(np.float64)
            ics[c0 + i] = _rank_ic_columns(v[:, None], np.asarray(ret[s:e], dtype=np.float64))[0]
    return ics


def run_arm(tr_packs, va_packs, va_days, ret_va, s_tr, s_va, K, mode, dev,
            n_sel, epochs=400, lr=0.05, y_scale=2.0):
    if mode == 'additive':
        w_tr = torch.ones(K, device=dev)
        w_va = torch.ones(K, device=dev)
        ic = _daily_ic(va_packs, w_va, K, ret_va, va_days)
        return float(np.nanmean(ic[:n_sel])), float(np.nanmean(ic[n_sel:])), ic, None
    p = torch.zeros(K, device=dev, requires_grad=True)

    def wf(s_day):
        if mode == 'static':
            return torch.softmax(p, -1) * K
        return torch.softmax(p.unsqueeze(0) * s_day.unsqueeze(1), -1) * K

    opt = torch.optim.Adam([p], lr=lr)
    best, best_p, since = -np.inf, p.detach().clone(), 0
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        _loss(tr_packs, wf(s_tr), K, y_scale).backward()
        opt.step()
        with torch.no_grad():
            ic = _daily_ic(va_packs, wf(s_va), K, ret_va, va_days)
        sel = float(np.nanmean(ic[:n_sel]))
        if sel > best + 1e-7:
            best, since, best_p = sel, 0, p.detach().clone()
        else:
            since += 1
            if since >= 25 and ep >= 30:
                break
    p.data.copy_(best_p)
    with torch.no_grad():
        ic = _daily_ic(va_packs, wf(s_va), K, ret_va, va_days)
    return (float(np.nanmean(ic[:n_sel])), float(np.nanmean(ic[n_sel:])), ic,
            p.detach().cpu().numpy().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(
        f'models/nam_gate/T115_idxrel_s{s}' for s in (42, 11, 23, 37)))
    ap.add_argument('--cache-dir', default='database/system_data/factors_cache_2026-08-18-idxrel')
    ap.add_argument('--group-map-file', default='scripts/exp/t118_group_map.json')
    ap.add_argument('--stocks', type=int, default=6000)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--select-holdout', type=float, default=0.4)
    ap.add_argument('--scalar-col', default='macro_m1m2_gap')
    ap.add_argument('--out', default='diagnose_output/T128_frozen_expert_gate.json')
    a = ap.parse_args()

    dirs = [d.strip() for d in a.models.split(',') if d.strip()]
    m0 = NAMGateModel()
    m0.load_model(os.path.join(dirs[0], 'nam_gate_factor_model.pkl'))
    feats = list(m0.feature_names)
    dev = torch.device(m0.device)
    print(f'冻结专家 {len(dirs)} 套，面板 {len(feats)} 列，device {dev}')

    end_dt = datetime.strptime(a.end, '%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * a.years)).strftime('%Y-%m-%d')
    TrainingConfig.FUTURE_DAYS = 7
    tr = MLModelTrainer(db_path=DATABASE_PATH, cache_dir=a.cache_dir)
    mgr = BaostockDataManager()
    codes = mgr.get_stock_list_from_db()['code'].tolist()[:a.stocks]
    mgr.close()
    sd = tr.load_label_data(codes, start, a.end)
    ds = tr.prepare_dataset(sd, train_start_date=start, train_end_date=a.end,
                            include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
                            n_jobs=4, target_features=None, use_factor_cache_only=True)
    regime = build_regime_matrix(DATABASE_PATH)
    fold = _prepare_fold(tr, ds, feats, 0.8, 1.0, regime, target='returns')
    del sd, ds

    tr_days, va_days = _day_slices(fold['d_train']), _day_slices(fold['d_val'])
    n_sel = max(1, int(round(len(va_days) * (1.0 - a.select_holdout))))
    si = m0.regime_cols.index(a.scalar_col)
    s_tr = torch.as_tensor(np.array([fold['M_train'][s][si] for s, _ in tr_days]),
                           dtype=torch.float32, device=dev)
    s_va = torch.as_tensor(np.array([fold['M_val'][s][si] for s, _ in va_days]),
                           dtype=torch.float32, device=dev)
    y_tr = torch.as_tensor(fold['y_train'], dtype=torch.float32, device=dev)
    ret_va = fold['ret_val']
    print(f'训练 {len(tr_days)} 日 / 验证 {len(va_days)} 日'
          f'（选型 {n_sel} / holdout {len(va_days) - n_sel}），门控标量 {a.scalar_col}')

    groupings = {'manual12': (np.asarray(m0.group_ids), len(m0.group_names))}
    if a.group_map_file and os.path.exists(a.group_map_file):
        gm = json.load(open(a.group_map_file, encoding='utf-8'))
        cids = sorted({int(v) for v in gm.values()})
        rm = {c: i for i, c in enumerate(cids)}
        groupings['clusters6'] = (np.array([rm[int(gm[f])] for f in feats], dtype=np.int64),
                                  len(cids))

    res = {}
    for d in dirs:
        seed = d.rsplit('_s', 1)[1]
        m = NAMGateModel()
        m.load_model(os.path.join(d, 'nam_gate_factor_model.pkl'))
        print(f'\n########## 冻结专家 = {d} ##########')
        for gname, (gid, K) in groupings.items():
            S_tr = precompute_group_sums(m, fold['X_train'], gid, K)
            S_va = precompute_group_sums(m, fold['X_val'], gid, K)
            for norm in (False, True):
                trp = pack_days(S_tr, tr_days, y_tr, norm=norm)
                vap = pack_days(S_va, va_days, None, norm=norm)
                for mode in ('additive', 'static', 'scalar'):
                    if mode == 'additive' and norm:
                        continue
                    sel, ho, ic, prm = run_arm(trp, vap, va_days, ret_va, s_tr, s_va,
                                               K, mode, dev, n_sel)
                    key = f'{gname}|{"norm" if norm else "raw"}|{mode}'
                    res.setdefault(key, {})[seed] = {
                        'sel': sel, 'holdout': ho, 'params': prm}
                    base = res.get(f'{gname}|raw|additive', {}).get(seed, {}).get('holdout')
                    tail = (f'  Δ vs additive={ho - base:+.5f}' if base is not None
                            and mode != 'additive' else '  ← 基线')
                    print(f'  s{seed} {key:30s} sel={sel:+.5f} holdout={ho:+.5f}{tail}')
                del trp, vap
            del S_tr, S_va
            torch.cuda.empty_cache()

    print('\n================ 汇总（配对 Δ vs 同一冻结专家的 additive）================')
    seeds = [d.rsplit('_s', 1)[1] for d in dirs]
    print(f'{"臂":32s} {"holdout均值":>11s} {"Δ均值":>9s} {"Δ中位":>9s} {"正种子":>7s} {"MDE":>8s}')
    summary = {}
    for key in res:
        g = key.split('|')[0]
        base = res[f'{g}|raw|additive']
        d = np.array([res[key][s]['holdout'] - base[s]['holdout'] for s in seeds])
        ho = np.array([res[key][s]['holdout'] for s in seeds])
        mde = 2 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        summary[key] = {'holdout_mean': float(ho.mean()), 'delta_mean': float(d.mean()),
                        'delta_median': float(np.median(d)),
                        'n_pos': int((d > 0).sum()), 'mde': float(mde)}
        print(f'{key:32s} {ho.mean():+11.5f} {d.mean():+9.5f} {np.median(d):+9.5f} '
              f'{int((d > 0).sum())}/{len(d):<5d} {mde:8.5f}')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump({'models': dirs, 'per_seed': res, 'summary': summary}, f,
                  ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    print('判读：additive 是同一套冻结专家下的门关基线（精确配对）。')
    print('      static 若拿走大部分增益 ⇒ 门的价值不是 regime 条件化，只是重分配族权重。')
    print('      晋级仍按 IC 优先协议：4/4 为正且 Δ中位 ≥0.005。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
