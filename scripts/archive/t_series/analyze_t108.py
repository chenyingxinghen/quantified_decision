"""T108 终审：全量池**门控**（`--lambda-lb 0.01` 防塌陷）vs 全量池**纯加性**，n=4 同种子配对。

判据预注册于 `$CLAUDE_JOB_DIR/tmp/t108.sh` 头部：
  ① 两窗都不输超过窗内 MDE（无 regime 债）；
  ② 合计 Δ > 合并 MDE ⇒ 门控进生产；
  ③ ① 不过 ⇒ s42 是运气，门控轴永久关闭。

**这一轮的特殊性**：IC 侧已明确劣化（s11 fp32 门控 0.09818 vs 纯加性 0.11114，
Δ −0.01395 ≈ 反向 2.3× 配对 MDE），所以门控**只能靠回测晋级**。
若它真的晋级，那本身是比「门控有效」更重要的结论 ——
说明**整截面 Rank IC 不是正确的载体判据**，必须写进台账并修正 IC 优先协议。
反过来，若不晋级，则 IC 与回测一致，协议无需改动。

⚠ 数值口径提示：s42/s11 的门控臂是 fp32 驻留，s23/s37 是 fp16（T110 等价性
PASS，Δ −0.00085 且早停同轮）。回测 MDE 是 30~50pp，1e-3 的输入扰动远在其下。

用法：python -u scripts/archive/t_series/analyze_t108.py
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
SEEDS = (42, 11, 23, 37)

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
    o = {k: (float(m.group(1)) if (m := r.search(txt)) else np.nan)
         for k, r in _RE.items()}
    m = _RE_BENCH.search(txt)
    o['bench'] = float(m.group(1)) if m else np.nan
    o['excess'] = o['ret'] - o['bench']
    return o


def main():
    data = {}
    for s in SEEDS:
        for win, _ in WINDOWS:
            # 门控臂：T107 用 s42、T108/T111 用 s11/s23/s37
            g = _read(f'T108_bt_gate_s{s}_{win}.log', f'T107_bt_gate_s{s}_{win}.log')
            # 纯加性基线：T099
            n = _read(f'T099_bt_full_s{s}_{win}.log')
            if g:
                data[('gate', s, win)] = _parse(g)
            if n:
                data[('add', s, win)] = _parse(n)

    stats = {}
    for win, wname in WINDOWS:
        print(f'\n=== {wname}市窗口 {win} ===')
        print(f"{'seed':>6} {'门控超额':>10} {'纯加性':>10} {'Δ(门−加)':>11}"
              f" {'门控β':>8} {'加性β':>8} {'门控胜率':>9} {'加性胜率':>9}")
        ds = []
        for s in SEEDS:
            g, a = data.get(('gate', s, win)), data.get(('add', s, win))
            if not (g and a):
                print(f'{s:>6}   （缺 {"门控" if not g else "纯加性"}）')
                continue
            d = g['excess'] - a['excess']
            ds.append(d)
            print(f"{s:>6} {g['excess']:>9.2f}pp {a['excess']:>9.2f}pp {d:>10.2f}pp"
                  f" {g['beta']:>8.3f} {a['beta']:>8.3f}"
                  f" {g['win']:>8.2f}% {a['win']:>8.2f}%")
        if len(ds) >= 2:
            ds = np.asarray(ds)
            mde = 2.0 * ds.std(ddof=1) / np.sqrt(len(ds))
            stats[win] = {'mean': float(ds.mean()), 'mde': float(mde), 'n': len(ds),
                          'wins': int((ds > 0).sum())}
            print(f'  Δ 均值 {ds.mean():+.2f}pp  中位 {np.median(ds):+.2f}pp'
                  f'  σ {ds.std(ddof=1):.2f}pp  MDE(n={len(ds)}) ≈ {mde:.2f}pp'
                  f'  {int((ds>0).sum())}/{len(ds)} 门控胜')

    if len(stats) < 2:
        print('\n窗口数据不齐，暂不裁决。')
        return

    print(f"\n{'='*72}\n判决（判据预注册于 t108.sh；IC 侧已劣化 −0.01395，只能靠回测晋级）\n{'='*72}")
    debts = [f'{wname}市输 {abs(stats[w]["mean"]):.2f}pp（> 窗内 MDE {stats[w]["mde"]:.2f}pp）'
             for w, wname in WINDOWS if stats[w]['mean'] < -stats[w]['mde']]
    if debts:
        print(f'**门1 不通过**：门控 {"、".join(debts)} → regime 债。')
        print('T107 的 s42 判定为单种子运气。**门控轴永久关闭**，生产维持纯加性全量池 NAM。')
        print('IC 与回测方向一致 ⇒ IC 优先协议无需修正。')
        return
    print('门1 通过：两窗都无超出噪声的落后。')

    tot = sum(stats[w]['mean'] for w, _ in WINDOWS)
    tot_mde = float(np.sqrt(sum(stats[w]['mde'] ** 2 for w, _ in WINDOWS)))
    if tot > tot_mde:
        print(f'**门2 通过**：两窗合计 Δ {tot:+.2f}pp > 合并 MDE {tot_mde:.2f}pp')
        print('\n**判决：门控（--lambda-lb 0.01）进生产。**')
        print('⚠ 这是台账首例「回测推翻 IC」：门控 holdout IC 低 0.01395（反向 2.3×MDE），'
              '却在两窗回测都不输且合计显著。必须记录：')
        print('  · 整截面 Rank IC 不是充分的载体判据 —— 它对头部 K=20 的重排不敏感；')
        print('  · 修正 [[alpha-not-rules-ic-first-protocol]]：IC 门槛可用于**关轴**'
              '（IC 涨但回测不涨 ⇒ 不晋级），不可用于**否决**已在回测上显著的臂；')
        print('  · 下一步须查清增益来自哪里：比对头部 K=20 的重叠率与 regime 分层胜率。')
        return
    print(f'门2 不通过：合计 Δ {tot:+.2f}pp ≤ 合并 MDE {tot_mde:.2f}pp → 门3 平局判据')

    def _avg(arm, key):
        vals = [data[(arm, s, w)][key] for s in SEEDS for w, _ in WINDOWS
                if (arm, s, w) in data]
        return float(np.mean(vals)) if vals else np.nan

    gb, ab = _avg('gate', 'beta'), _avg('add', 'beta')
    gd, ad = _avg('gate', 'dd'), _avg('add', 'dd')
    print(f'  平均 β：门控 {gb:.3f} vs 纯加性 {ab:.3f}（越接近 1 越好）')
    print(f'  平均最大回撤：门控 {gd:.2f}% vs 纯加性 {ad:.2f}%（越浅越好）')
    beta_better = abs(gb - 1) < abs(ab - 1)
    dd_better = gd > ad
    if beta_better == dd_better:
        w = '门控' if beta_better else '纯加性'
    else:
        w = '门控' if beta_better else '纯加性'
        print('  （β 与回撤指向不一致，按 β 判 —— 与 T095/T105 同口径）')
    print(f'\n**判决：{w} 进生产（平局判据）。**')
    if w == '纯加性':
        print('门控在两窗都没输出噪声外的差距，但也没赢 —— 加了 11 个门控参数换来平局，'
              '按奥卡姆取更简的纯加性。门控轴关闭。')
    else:
        print('⚠ 平局判据选中门控，但合计 Δ 未超 MDE ⇒ 证据强度弱于 T098 那种 7×MDE。'
              '进生产前应补种子把 n 提到 6 以上。')


if __name__ == '__main__':
    main()
