import os, re, numpy as np
D = 'diagnose_output'
RE = {'ret': re.compile(r'总收益率:\s*(-?[\d.]+)%'), 'dd': re.compile(r'最大回撤:\s*(-?[\d.]+)%'),
      'sharpe': re.compile(r'夏普比率:\s*(-?[\d.]+)'), 'beta': re.compile(r'beta\s*:\s*(-?[\d.]+)')}
RB = re.compile(r'全市场等权\(日频再平衡\)\s+(-?[\d.]+)%')

def rd(arm, s, w, pref):
    for d in (D, os.path.join(D, 'iter_output')):
        p = os.path.join(d, f'{pref}_bt_{arm}_s{s}_{w}.log')
        if os.path.isfile(p):
            return open(p, encoding='utf-8', errors='replace').read()

def pa(t):
    o = {}
    for k, r in RE.items():
        m = r.search(t)
        o[k] = float(m.group(1)) if m else np.nan
    m = RB.search(t)
    o['bench'] = float(m.group(1)) if m else np.nan
    o['excess'] = o['ret'] - o['bench']
    return o

for w, name in (('2022-09-05', '熊'), ('2024-08-05', '牛')):
    print(f'\n=== {name}市 {w} ===')
    print(f"{'seed':>5} {'全量超额':>11} {'800只超额':>11} {'Δ':>9} {'全量β':>8} {'800β':>7} {'全量回撤':>9} {'全量夏普':>8}")
    ds = []
    for s in (42, 11, 23, 37):
        a = rd('full', s, w, 'T099')
        if not a:
            print(f'{s:>5}   （未完成）')
            continue
        A = pa(a)
        b = rd('nam', s, w, 'T095')
        B = pa(b) if b else None
        bx = f"{B['excess']:.2f}pp" if B else '—'
        bb = f"{B['beta']:.3f}" if B else '—'
        dv = A['excess'] - B['excess'] if B else float('nan')
        if B:
            ds.append(dv)
        print(f"{s:>5} {A['excess']:>10.2f}pp {bx:>11} {dv:>8.2f}pp "
              f"{A['beta']:>8.3f} {bb:>7} {A['dd']:>8.2f}% {A['sharpe']:>8.3f}")
    if len(ds) >= 2:
        ds = np.asarray(ds)
        print(f"  配对 Δ 均值 {ds.mean():+.2f}pp  σ {ds.std(ddof=1):.2f}pp  "
              f"MDE≈{2.0*ds.std(ddof=1)/np.sqrt(len(ds)):.2f}pp  {int((ds>0).sum())}/{len(ds)} 全量胜")
