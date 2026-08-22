"""T082 / E12：IC 的**可迁移性**诊断 —— 零训练，只做前向。

为什么要做这个
--------------
T078 立的 IC-first 协议有一条隐含假设：**验证窗 Rank IC ↑ ⇒ OOS 收益 ↑**。
T081 是这个假设的第一次确认性检验，结果打脸：多持有期标签在验证窗
（2020-04 → 2022-09）Rank IC 4/4 涨、top5 超额 3/4 涨，但 OOS 熊市
（2022-09 → 2024-08）回测超额 4/4 掉 ~9pp。执行层已排除（两臂 stop_loss 占比
17~20%、平均持有 6.9 天，几乎完全一致），所以断裂点在**选股质量随行情漂移**。

本脚本用同一批已保存的模型（不重训）回答两个问题：

  Q1（决定协议怎么改）验证窗内把交易日按行情分层，多期标签的 IC 增益是不是
      **只长在上涨日**？若是 ⇒ 只要把门槛改成「上涨日和下跌日都不劣化」，
      E11 这类 regime 押注在**样本内**就会被拦下，不必再花 8 次回测 + 8 次零假设。
  Q2（验证 Q1 的结论确实指向 OOS）同一批模型在 OOS 熊/牛窗的日度 IC 与 top-K 超额，
      是否与回测 z 同向？OOS 窗**只作证据、不作筛选**——用 OOS 挑轴等于把
      确认集烧成训练集，那是自欺。

口径
----
* 打分路径与训练/回测一致：``(X - input_mean) / input_std`` 后走 ``net.forward_day``。
  本台账的模型都是 ``--disable-gate``，门控是恒等，但仍按日传 regime 行以防口径漂移。
* 日度指标：Rank IC（打分 vs 7 日前向收益）、top-K 超额（前 K 名平均前向收益 −
  当日全截面平均），K 默认 40，与 K=40 回测口径对齐。
* 分层：① 全部；② 当日全截面平均前向收益 > 0 / < 0（「这 7 天市场涨没涨」）；
  ③ 按自然年。②是 Q1 的主判层。

用法
----
  python scripts/exp/diag_ic_by_regime.py \
      --model base_s42=models/nam_gate/T045 \
      --model mh_s42=models/nam_gate/T081_mh_s42 \
      --years 13 --end 2026-08-05 --out diagnose_output/T082_ic_by_regime.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.baostock_config import DATABASE_PATH  # noqa: E402
from config.factor_config import TrainingConfig  # noqa: E402
from core.data.baostock_main import BaostockDataManager  # noqa: E402
from core.factors.nam_gate_model import NAMGateModel  # noqa: E402
from core.factors.regime_features import build_regime_matrix  # noqa: E402
from core.factors.train_ml_model import MLModelTrainer  # noqa: E402
from scripts.train_nam_model import _prepare_fold, _day_slices  # noqa: E402

# 三段窗口：W0 是训练时用于早停的验证窗（样本内），W1/W2 是回测用的 OOS 窗。
WINDOWS = [
    ('val_insample', '2020-04-07', '2022-09-05'),
    ('oos_bear', '2022-09-05', '2024-08-05'),
    ('oos_bull', '2024-08-05', '2026-08-05'),
]


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    xv, yv = x[m], y[m]
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return np.nan
    return float(np.corrcoef(rankdata(xv), rankdata(yv))[0, 1])


def _topk_excess(score: np.ndarray, ret: np.ndarray, k: int) -> float:
    m = np.isfinite(score) & np.isfinite(ret)
    if m.sum() < k * 2:
        return np.nan
    s, r = score[m], ret[m]
    idx = np.argsort(-s)[:k]
    return float(r[idx].mean() - r.mean())


def score_daily(model: NAMGateModel, X, ret, dates, M, topk: int):
    """逐日打分 → (日期, Rank IC, topK 超额, 当日截面平均收益)。"""
    Xn = X.astype(np.float32)
    if model.input_mean is not None and model.input_std is not None:
        Xn = ((Xn - np.asarray(model.input_mean, dtype=np.float32))
              / np.asarray(model.input_std, dtype=np.float32)).astype(np.float32)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)

    dev = torch.device(model.device)
    net = model.net
    net.eval()
    day, ics, tex, mkt = [], [], [], []
    with torch.no_grad():
        for (s, e) in _day_slices(dates):
            if e - s < max(20, topk * 2):
                continue
            xt = torch.as_tensor(Xn[s:e], dtype=torch.float32, device=dev)
            mt = torch.as_tensor(np.asarray(M[s], dtype=np.float32),
                                 dtype=torch.float32, device=dev)
            sc, _, _ = net.forward_day(xt, mt)
            sc = sc.cpu().numpy().astype(np.float64)
            y = np.asarray(ret[s:e], dtype=np.float64)
            day.append(dates[s])
            ics.append(_rank_ic(sc, y))
            tex.append(_topk_excess(sc, y, topk))
            mkt.append(float(np.nanmean(y)))
    return (np.array(day), np.array(ics, dtype=np.float64),
            np.array(tex, dtype=np.float64), np.array(mkt, dtype=np.float64))


def _agg(ic, tex, mask):
    """一层的聚合结果；日数不足 20 视为不可判。"""
    n = int(np.sum(mask & np.isfinite(ic)))
    if n < 20:
        return {'n_days': n, 'ic': None, 'topk_excess': None}
    return {
        'n_days': n,
        'ic': float(np.nanmean(ic[mask])),
        'topk_excess': float(np.nanmean(tex[mask])),
    }


# 头部越细，交易时越接近真实选股比例：生产是 ~5000 选 40，即 top 0.8%。
HEAD_CUTS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50)


def head_profile(model: NAMGateModel, X, ret, dates, M):
    """T085：打分**极端头部**还有没有分辨力。

    每档给两个量：该档平均前向收益的**当日截面分位**（0.5=无信息），以及相对
    当日均值的超额。若 top0.2% 不优于 top20%，说明打分在真正交易的那一档上没有
    增量信息——那时该动的是**损失函数的头部权重**，不是再找特征。
    """
    Xn = X.astype(np.float32)
    if model.input_mean is not None and model.input_std is not None:
        Xn = ((Xn - np.asarray(model.input_mean, dtype=np.float32))
              / np.asarray(model.input_std, dtype=np.float32)).astype(np.float32)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    dev = torch.device(model.device)
    net = model.net
    net.eval()
    acc = {c: [[], []] for c in HEAD_CUTS}
    with torch.no_grad():
        for (s, e) in _day_slices(dates):
            n = e - s
            if n < 200:                       # 档位太细需要足够宽的截面
                continue
            xt = torch.as_tensor(Xn[s:e], dtype=torch.float32, device=dev)
            mt = torch.as_tensor(np.asarray(M[s], dtype=np.float32),
                                 dtype=torch.float32, device=dev)
            sc, _, _ = net.forward_day(xt, mt)
            sc = sc.cpu().numpy().astype(np.float64)
            y = np.asarray(ret[s:e], dtype=np.float64)
            ok = np.isfinite(sc) & np.isfinite(y)
            if ok.sum() < 200:
                continue
            sc, y = sc[ok], y[ok]
            ypct = (rankdata(y) - 0.5) / len(y)
            order = np.argsort(-sc)
            for c in HEAD_CUTS:
                kk = max(3, int(round(len(sc) * c)))
                idx = order[:kk]
                acc[c][0].append(float(ypct[idx].mean()))
                acc[c][1].append(float(y[idx].mean() - y.mean()))
    return {f'top{c}': {'ret_pct': float(np.mean(v[0])) if v[0] else None,
                        'excess': float(np.mean(v[1])) if v[1] else None,
                        'n_days': len(v[0])}
            for c, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', required=True,
                    help='NAME=DIR 或 NAME=PKL，可重复；一次数据加载评估全部模型')
    ap.add_argument('--stocks', type=int, default=800)
    ap.add_argument('--code-mode', default='head',
                    choices=['head', 'holdout_random', 'random'],
                    help='股票池取法。head=codes[:N]（训练用的口径，只有 000/001/002 前缀）；'
                         'holdout_random=从 codes[N:] 里随机取 N 只（训练完全没见过的板块）；'
                         'random=全市场随机取 N 只。T084：回测 85%% 的成交落在训练池之外，'
                         '所以要看模型在没见过的股票上排得准不准')
    ap.add_argument('--code-seed', type=int, default=20260813)
    ap.add_argument('--head-profile', action='store_true',
                    help='T085：额外输出 top0.2%%~50%% 各档的实现收益分位与超额，'
                         '看极端头部（生产真正交易的那一档）还有没有分辨力')
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2026-08-05')
    ap.add_argument('--topk', type=int, default=40)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='diagnose_output/T082_ic_by_regime.json')
    args = ap.parse_args()

    specs = []
    for item in args.model:
        name, _, path = item.partition('=')
        if not path:
            raise SystemExit(f'--model 需 NAME=PATH 形式，收到 {item!r}')
        if os.path.isdir(path):
            path = os.path.join(path, 'nam_gate_factor_model.pkl')
        if not os.path.isfile(path):
            raise SystemExit(f'模型不存在: {path}')
        specs.append((name, path))

    models = {}
    feat_ref = None
    for name, path in specs:
        m = NAMGateModel()
        m.load_model(path)
        if m.input_mean is None:
            raise SystemExit(f'{name} 缺归一化统计量（T043 铁律），不可评估')
        names = list(m.feature_names)
        if feat_ref is None:
            feat_ref = names
        elif names != feat_ref:
            raise SystemExit(f'{name} 的特征列与首个模型不一致，无法同口径对比')
        models[name] = m
        print(f'[LOAD] {name}: {len(names)} 特征 / K={len(m.group_names)}')

    end_dt = datetime.strptime(args.end, '%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * args.years)).strftime('%Y-%m-%d')
    end = end_dt.strftime('%Y-%m-%d')
    print(f'[DATA] {start} → {end}, {args.stocks} 只')

    TrainingConfig.FUTURE_DAYS = 7
    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    all_codes = manager.get_stock_list_from_db()['code'].tolist()
    manager.close()
    if args.code_mode == 'head':
        codes = all_codes[:args.stocks]
    else:
        pool = all_codes[args.stocks:] if args.code_mode == 'holdout_random' else all_codes
        rng = np.random.default_rng(args.code_seed)
        idx = rng.choice(len(pool), size=min(args.stocks, len(pool)), replace=False)
        codes = sorted(pool[i] for i in idx)
    print(f'[UNIV] mode={args.code_mode} n={len(codes)} '
          f'前缀 {pd.Series([c[:3] for c in codes]).value_counts().head(6).to_dict()}')
    stocks_data = trainer.load_label_data(codes, start, end)
    dataset = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=args.workers, target_features=None, use_factor_cache_only=True,
        return_sample_metadata=False,
    )
    regime = build_regime_matrix(DATABASE_PATH)
    # 切 0.8 只是为了复用 _prepare_fold 的列选择/对齐；两段拼回去就是全样本面板。
    fold = _prepare_fold(trainer, dataset, feat_ref, 0.8, 1.0, regime, target='returns')
    X = np.concatenate([fold['X_train'], fold['X_val']], axis=0)
    ret = np.concatenate([fold['ret_train'], fold['ret_val']], axis=0)
    dates = np.concatenate([fold['d_train'], fold['d_val']], axis=0)
    M = np.concatenate([fold['M_train'], fold['M_val']], axis=0)
    order = np.argsort(dates, kind='stable')
    X, ret, dates, M = X[order], ret[order], dates[order], M[order]
    print(f'[PANEL] {len(X)} 样本 / {dates[0]} → {dates[-1]}')

    per_model = {}
    for name, m in models.items():
        d, ic, tex, mkt = score_daily(m, X, ret, dates, M, args.topk)
        ds = pd.to_datetime(pd.Series(d))
        res = {}
        for wname, ws, we in WINDOWS:
            inwin = ((ds >= ws) & (ds < we)).to_numpy()
            if inwin.sum() == 0:
                continue
            up = inwin & (mkt > 0)
            dn = inwin & (mkt <= 0)
            res[wname] = {
                'all': _agg(ic, tex, inwin),
                'up_days': _agg(ic, tex, up),
                'down_days': _agg(ic, tex, dn),
                'by_year': {str(y): _agg(ic, tex, inwin & (ds.dt.year == y).to_numpy())
                            for y in sorted(ds[inwin].dt.year.unique())},
            }
        if args.head_profile:
            sel = pd.to_datetime(pd.Series(dates))
            for wname, ws, we in WINDOWS:
                if wname not in res:
                    continue
                mk = ((sel >= ws) & (sel < we)).to_numpy()
                if mk.sum() == 0:
                    continue
                res[wname]['head'] = head_profile(m, X[mk], ret[mk], dates[mk], M[mk])
        per_model[name] = res
        a = res.get('val_insample', {}).get('all', {})
        print(f'[SCORED] {name}: val_insample IC='
              f'{a.get("ic")!r} days={a.get("n_days")}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'topk': args.topk, 'windows': WINDOWS, 'models': per_model},
                  f, ensure_ascii=False, indent=1)
    print(f'[OUT] {args.out}')


if __name__ == '__main__':
    main()
