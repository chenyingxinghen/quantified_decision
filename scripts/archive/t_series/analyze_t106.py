"""T106 判决：全量池 + 门控 + lambda_lb 防塌陷 vs 全量池纯加性基线。

两道判据（预注册于 run_t106_gate_lambda.sh），**门控健康度先于 IC**：

  ① 健康：max_share < 0.60（代码 809 行的内置阈值）且门控权重的跨日变异系数
     CV > 0.05。T102 的教训是 status 族占比 98.4% 且 **CV 只有 0.0025** ——
     那是静态重加权，与 regime 无关，`w_k·f_k` 尺度互相抵消的重参数化而已。
     门控不健康时，IC 涨了也不能算「门控有效」。
  ② IC：Δ holdout ≥ +0.005 值得扩种子；|Δ| < 0.002 关轴。

用法：python -u scripts/archive/t_series/analyze_t106.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT)
FOLD = '80-100%'
ARMS = (('base(纯加性)', 'T098_full_s42', None),
        ('lb001', 'T106_lb001_s42', 'nam_gate_T106_lb001'),
        ('lb01', 'T106_lb01_s42', 'nam_gate_T106_lb01'))


def _load(stem):
    for d in (os.path.join(ROOT, 'diagnose_output'),
              os.path.join(ROOT, 'diagnose_output', 'iter_output')):
        p = os.path.join(d, f'{stem}.json')
        if os.path.isfile(p):
            f = json.load(open(p, encoding='utf-8'))['folds'].get(FOLD)
            if f and 'nam_gate' in f:
                return f['nam_gate']
    return None


def _gate_cv(plot_dir):
    """门控权重的跨日变异系数 —— 区分「regime 路由」与「静态重加权」的关键量。"""
    if not plot_dir:
        return None
    p = os.path.join(ROOT, 'diagnose_output', plot_dir, 'gate_weights.csv')
    if not os.path.isfile(p):
        return None
    g = pd.read_csv(p, index_col=0)
    top = g.mean(axis=0).idxmax()
    col = g[top]
    return {'top_group': top, 'cv': float(col.std() / max(abs(col.mean()), 1e-9)),
            'mean_share': float(g.mean(axis=0).max() / g.mean(axis=0).sum())}


def main():
    base = _load('T098_full_s42')
    print('全量池 5480 只 / 折 80-100% / 单种子 42 / 无偏 holdout\n')
    print(f"{'臂':>14} {'holdout':>9} {'Δ vs base':>11} {'跌日IC':>9} "
          f"{'门控熵':>7} {'最大占比':>9} {'主族CV':>8}")
    rows = {}
    for name, stem, pdir in ARMS:
        r = _load(stem)
        if not r:
            print(f'{name:>14}   （尚无结果）')
            continue
        rows[name] = r
        cv = _gate_cv(pdir)
        d = '' if base is None or stem == 'T098_full_s42' else \
            f"{r['rank_ic_holdout'] - base['rank_ic_holdout']:+11.5f}"
        print(f"{name:>14} {r['rank_ic_holdout']:>9.5f} {d:>11} "
              f"{r['regime']['down_days']['rank_ic']:>9.5f} "
              f"{r.get('gate_entropy', float('nan')):>7.3f} "
              f"{r.get('gate_max_share', float('nan')):>9.3f} "
              f"{(f'{cv[chr(99)+chr(118)]:.4f}' if cv else '—'):>8}")

    print(f"\n门控熵上界 ln(11)={np.log(11):.3f}；均匀时 max_share=0.091")
    print('判据①健康：max_share < 0.60 且主族 CV > 0.05（CV≈0 ⇒ 静态重加权，非 regime 路由）')
    print('判据②IC：Δ ≥ +0.005 扩种子；|Δ| < 0.002 关轴\n')

    if base is None:
        print('缺 T098_full_s42 基线，无法判 IC。')
        return
    for name, stem, pdir in ARMS[1:]:
        r = rows.get(name)
        if not r:
            continue
        d = r['rank_ic_holdout'] - base['rank_ic_holdout']
        cv = _gate_cv(pdir)
        share = r.get('gate_max_share', 1.0)
        healthy = share < 0.60 and (cv is not None and cv['cv'] > 0.05)
        ic_v = ('扩种子' if d >= 0.005 else '关轴' if abs(d) < 0.002 else '灰区')
        print(f'  {name}: Δ {d:+.5f} → IC 判 **{ic_v}**；'
              f'门控 {"健康" if healthy else "不健康"}'
              + ('' if cv is None else f'（主族 {cv["top_group"]}，占比 {cv["mean_share"]:.3f}，'
                                       f'CV {cv["cv"]:.4f}）'))
        if not healthy and d >= 0.005:
            print('    ⚠ IC 上升但门控不健康 —— 不能归因于「学到 regime 路由」，'
                  '更可能是重参数化带来的方差。需要多种子才能分辨。')


if __name__ == '__main__':
    main()
