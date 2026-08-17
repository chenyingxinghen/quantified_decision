"""T094：截面排名等权混合（xgb+lgb）vs 单模型 vs NAM 的 IC 终审表。

**为什么是这个比法**：T092 证明 xgb/lgb 学到的是两个极端的风格暴露（gain 重要性
Spearman ρ 仅 0.457，market_cap 占比 17.2% vs 1.1%），而 T094 的 oracle 诊断证明
「何时该用谁」不可预测（折外 R²=−0.19、月均 ΔIC 离散度已低于纯噪声理论值）。
条件混合关轴后，剩下唯一值得验的是**静态**等权混合 —— 它吃的是分散化，不是择时。

**口径**：
- 主读数用 ``p_fix``（固定 300 棵、无验证集选型），三个臂都无偏，可直接横比。
- 混合 = 各自当日截面 rank(pct) 后 0.5/0.5 平均。必须先排名再平均：两个模型的
  分数量纲不同（XGBRanker 的 margin vs lambdarank 的 raw score），直接平均等于
  按方差隐式加权，[[ensemble-beats-single]] 记的「权重必须 0.5/0.5」也是这个前提。
- 行情分层沿用 ``_regime_stratified_metrics`` 的定义（当日全截面平均前向收益符号），
  与 T082/E12 门槛同口径。

用法：
  python -u scripts/archive/t_series/analyze_blend.py --seeds 42,11,23,37
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

FOLDS = ('60-80', '70-90', '80-100')


def _day_slices(dates):
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    return [(int(s), int(s + c)) for s, c in zip(starts, counts)]


def _rank_pct(p, slices):
    out = np.empty(len(p), dtype=np.float64)
    for s, e in slices:
        out[s:e] = rankdata(p[s:e]) / (e - s)
    return out


def _metrics(pred, ret, slices, k=40):
    """折内逐日 IC / Top-K 超额，按当日市场涨跌分层。"""
    ic, ex, mkt = [], [], []
    for s, e in slices:
        n = e - s
        if n < 10:
            continue
        p, r = np.asarray(pred[s:e], np.float64), np.asarray(ret[s:e], np.float64)
        v = np.corrcoef(rankdata(p), rankdata(r))[0, 1]
        if not np.isfinite(v):
            continue
        kk = min(k, n)
        top = np.argpartition(p, -kk)[-kk:]
        ic.append(float(v))
        ex.append(float(r[top].mean() - r.mean()))
        mkt.append(float(r.mean()))
    ic, ex, up = np.asarray(ic), np.asarray(ex), np.asarray(mkt) > 0
    out = {}
    for name, m in (('all', np.ones_like(up)), ('up', up), ('down', ~up)):
        out[name] = (float(ic[m].mean()), float(ex[m].mean()), int(m.sum())) \
            if m.sum() >= 20 else (np.nan, np.nan, int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,11,23,37')
    ap.add_argument('--prefix', default='diagnose_output/T094_preds')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    rows = []
    for seed in seeds:
        for tag in FOLDS:
            fx = f'{args.prefix}_xgb_s{seed}_{tag}.npz'
            fl = f'{args.prefix}_lgb_s{seed}_{tag}.npz'
            if not (os.path.exists(fx) and os.path.exists(fl)):
                print(f'  [缺] seed={seed} 折={tag}')
                continue
            x, l = np.load(fx), np.load(fl)
            assert np.array_equal(x['dates'], l['dates']), '两模型的验证日不一致'
            assert np.allclose(x['ret'], l['ret']), '两模型的前向收益不一致'
            sl = _day_slices(x['dates'])
            ret = x['ret']
            arms = {
                'xgb': x['p_fix'],
                'lgb': l['p_fix'],
                'blend': 0.5 * _rank_pct(x['p_fix'], sl) + 0.5 * _rank_pct(l['p_fix'], sl),
            }
            for arm, p in arms.items():
                m = _metrics(p, ret, sl)
                rows.append({'seed': seed, 'fold': tag, 'arm': arm,
                             'ic': m['all'][0], 'ic_up': m['up'][0], 'ic_down': m['down'][0],
                             'ex': m['all'][1], 'ex_up': m['up'][1], 'ex_down': m['down'][1],
                             'n_down': m['down'][2]})
    df = pd.DataFrame(rows)
    if df.empty:
        print('没有可用的预测落盘')
        return

    print('\n=== 折均 IC（无偏口径 p_fix）===')
    piv = df.pivot_table(index='seed', columns='arm', values='ic', aggfunc='mean')
    piv = piv[[c for c in ('xgb', 'lgb', 'blend') if c in piv.columns]]
    print(piv.to_string(float_format=lambda v: f'{v:.5f}'))
    print(f"\n  跨种子均值: " + '  '.join(f'{c}={piv[c].mean():.5f}' for c in piv.columns))
    print(f"  σ_seed:     " + '  '.join(f'{c}={piv[c].std(ddof=1):.5f}' for c in piv.columns))

    if {'blend', 'xgb', 'lgb'} <= set(piv.columns):
        # 三组对照，判定看第一组：
        #   vs lgb  —— **现役生产模型**，这才是「该不该换」的那个比较
        #   vs xgb  —— 另一个成分，做旁证
        #   vs max  —— 逐格用后视镜挑更强的那个当基准，是刻意苛刻的上界对照，
        #             不作判据（真实交易时无法预知哪折哪个模型更强）
        w = df.pivot_table(index=['seed', 'fold'], columns='arm',
                           values=['ic_up', 'ic_down'], aggfunc='mean')
        for base, label, is_gate in (('lgb', '现役 LightGBM', True),
                                     ('xgb', 'XGBoost', False),
                                     (None, '逐格更强单模型（后视镜上界）', False)):
            if base is None:
                bs = piv[['xgb', 'lgb']].max(axis=1)
                bu = w[[('ic_up', 'xgb'), ('ic_up', 'lgb')]].max(axis=1)
                bd = w[[('ic_down', 'xgb'), ('ic_down', 'lgb')]].max(axis=1)
            else:
                bs, bu, bd = piv[base], w[('ic_up', base)], w[('ic_down', base)]
            d, du, dd = piv['blend'] - bs, w[('ic_up', 'blend')] - bu, w[('ic_down', 'blend')] - bd
            mark = '【判据】' if is_gate else '（旁证）'
            print(f'\n=== {mark}混合 − {label} ===')
            for s in piv.index:
                print(f'  s{s}: {piv.loc[s, "blend"]:.5f} vs {bs[s]:.5f}   Δ {d[s]:+.5f}')
            print(f'  折均 IC: {int((d > 0).sum())}/{len(d)} 上升，Δ 中位 {d.median():+.5f}')
            print(f'  up_days    {int((du > 0).sum())}/{len(du)} 正向，中位 {du.median():+.5f}')
            print(f'  down_days  {int((dd > 0).sum())}/{len(dd)} 正向，中位 {dd.median():+.5f}')
            if is_gate:
                g1 = int((d > 0).sum()) == len(d)
                g2 = (dd > 0).sum() > len(dd) / 3
                print(f'  → 4/4 同向: {"通过" if g1 else "**不通过**"}   '
                      f'下跌日 >1/3 正向: {"通过" if g2 else "**不通过**"}')

    print('\n=== 逐折明细 ===')
    print(df.pivot_table(index=['fold'], columns='arm', values='ic', aggfunc='mean')
          .to_string(float_format=lambda v: f'{v:.5f}'))


if __name__ == '__main__':
    main()
