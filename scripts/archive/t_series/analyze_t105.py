"""T105 终审：全量池 blend（xgb+lgb 等权）vs 全量池 NAM，n=3 同种子号对照。

判据预注册在 run_t105_blend_final.sh 头部（T095 三门）。blend 数据来自
T103（s42）+ T105（s11/s23），NAM 来自 T099（取同种子号 42/11/23）。

⚠ 种子号跨模型族没有因果联系，「配对」只是沿用 T095 口径 ——
σ_paired ≈ √(σ_blend² + σ_nam²)，MDE 按这个成分理解。

用法：python -u scripts/archive/t_series/analyze_t105.py
"""

import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT)
DIRS = (os.path.join(ROOT, 'diagnose_output'),
        os.path.join(ROOT, 'diagnose_output', 'iter_output'))
WINDOWS = (('2022-09-05', '熊'), ('2024-08-05', '牛'))
SEEDS = (42, 11, 23)

_RE = {'ret': re.compile(r'总收益率:\s*(-?[\d.]+)%'),
       'dd': re.compile(r'最大回撤:\s*(-?[\d.]+)%'),
       'sharpe': re.compile(r'夏普比率:\s*(-?[\d.]+)'),
       'beta': re.compile(r'beta\s*:\s*(-?[\d.]+)'),
       'win': re.compile(r'盈利交易:\s*\d+\s*\(([\d.]+)%\)')}
_RE_BENCH = re.compile(r'全市场等权\(日频再平衡\)\s+(-?[\d.]+)%')


def _read(*names):
    for name in names:
        for d in DIRS:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return open(p, encoding='utf-8', errors='replace').read()
    return None


def _parse(txt):
    o = {}
    for k, r in _RE.items():
        m = r.search(txt)
        o[k] = float(m.group(1)) if m else np.nan
    m = _RE_BENCH.search(txt)
    o['bench'] = float(m.group(1)) if m else np.nan
    o['excess'] = o['ret'] - o['bench']
    return o


def main():
    data = {}
    for s in SEEDS:
        for win, _ in WINDOWS:
            b = _read(f'T105_bt_blend_s{s}_{win}.log', f'T103_bt_blend_s{s}_{win}.log')
            n = _read(f'T099_bt_full_s{s}_{win}.log')
            if b:
                data[('blend', s, win)] = _parse(b)
            if n:
                data[('nam', s, win)] = _parse(n)

    stats = {}
    for win, wname in WINDOWS:
        print(f'\n=== {wname}市窗口 {win} ===')
        print(f"{'seed':>6} {'blend超额':>10} {'nam超额':>10} {'Δ(b−n)':>10}"
              f" {'blend β':>9} {'nam β':>8} {'blend回撤':>10} {'nam回撤':>9}")
        ds = []
        for s in SEEDS:
            b, n = data.get(('blend', s, win)), data.get(('nam', s, win))
            if not (b and n):
                print(f'{s:>6}   （缺 {"blend" if not b else "nam"}）')
                continue
            d = b['excess'] - n['excess']
            ds.append(d)
            print(f"{s:>6} {b['excess']:>9.2f}pp {n['excess']:>9.2f}pp {d:>9.2f}pp"
                  f" {b['beta']:>9.3f} {n['beta']:>8.3f} {b['dd']:>9.2f}% {n['dd']:>8.2f}%")
        if len(ds) >= 2:
            ds = np.asarray(ds)
            mde = 2.0 * ds.std(ddof=1) / np.sqrt(len(ds))
            stats[win] = {'mean': ds.mean(), 'mde': mde, 'n': len(ds),
                          'wins': int((ds > 0).sum())}
            print(f'  Δ 均值 {ds.mean():+.2f}pp  σ {ds.std(ddof=1):.2f}pp'
                  f'  MDE(n={len(ds)}) ≈ {mde:.2f}pp  {int((ds>0).sum())}/{len(ds)} blend 胜')

    if len(stats) < 2:
        print('\n窗口数据不齐，暂不裁决。')
        return

    print(f"\n{'='*70}\n判决（判据预注册于 run_t105_blend_final.sh）\n{'='*70}")
    debts = []
    for win, wname in WINDOWS:
        st = stats[win]
        if st['mean'] < -st['mde']:
            debts.append(f'{wname}市输 {abs(st["mean"]):.2f}pp（> MDE {st["mde"]:.2f}pp）')
    if debts:
        print(f'门1 不通过：blend {"、".join(debts)} → regime 债，blend 出局，'
              f'生产维持 NAM，混合轴永久关闭。')
        return
    print('门1 通过：两窗都无超出噪声的落后。')

    tot = sum(stats[w]['mean'] for w, _ in WINDOWS)
    tot_mde = float(np.sqrt(sum(stats[w]['mde'] ** 2 for w, _ in WINDOWS)))
    if tot > tot_mde:
        print(f'门2 通过：两窗合计 Δ {tot:+.2f}pp > 合并 MDE {tot_mde:.2f}pp')
        print('\n**判决：blend（全量池 xgb+lgb 等权）进生产。**')
        return
    print(f'门2 不通过：合计 Δ {tot:+.2f}pp ≤ 合并 MDE {tot_mde:.2f}pp → 门3 平局判据')

    def _avg(arm, key):
        vals = [data[(arm, s, w)][key] for s in SEEDS for w, _ in WINDOWS
                if (arm, s, w) in data]
        return float(np.mean(vals)) if vals else np.nan

    bb, nb = _avg('blend', 'beta'), _avg('nam', 'beta')
    bd, nd = _avg('blend', 'dd'), _avg('nam', 'dd')
    print(f'  平均 β：blend {bb:.3f} vs nam {nb:.3f}（越接近 1 越好）')
    print(f'  平均最大回撤：blend {bd:.2f}% vs nam {nd:.2f}%（越浅越好）')
    b_beta_better = abs(bb - 1) < abs(nb - 1)
    b_dd_better = bd > nd  # dd 是负数，越大（越接近 0）越浅
    if b_beta_better and b_dd_better:
        w = 'blend'
    elif (not b_beta_better) and (not b_dd_better):
        w = 'NAM'
    else:
        w = 'blend' if b_beta_better else 'NAM'
        print('  （β 与回撤指向不一致，按 β 判 —— 与 T095 同口径）')
    print(f'\n**判决：{w} 进生产（平局判据）。**')


if __name__ == '__main__':
    main()
