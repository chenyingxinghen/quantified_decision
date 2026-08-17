"""T095 终审判决：xgb+lgb 等权混合 vs NAM，4 种子配对 × 2 个 regime 窗口。

判据（2026-08-15 用户预注册）：
  1. **无 regime 债**：两个窗口都不输对手超过窗内 MDE。一窗大赢一窗大输 =
     风格彩票，直接出局 —— 不管两窗合计多好看。
     这条是把 [[regime-stratified-ic-gate]] 的下跌日逻辑上移到回测层：
     单窗口胜负由测试窗的 regime 组成决定，而 T094 已证明没人能预测下一段是牛是熊。
  2. **有净增益**：满足 1 的前提下，两窗合计超额为正且过噪声门。
  3. 平局判给 β 更接近 1、回撤更浅的那个（不靠风格站位的钱优先）。

MDE 用**实测**跨种子离散度算，不用台账里的历史常数：
配对 n=4 的 MDE ≈ 2.0 × σ_paired / √4（双侧 α=0.05、power≈0.6 的粗尺度，
与 [[multifold-ic-evaluator-mde.md]] 同一套算法），σ_paired 取逐种子配对差的标准差。

用法：python -u scripts/archive/t_series/analyze_t095.py
"""

import os
import re
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

SEEDS = (42, 11, 23, 37)
WINDOWS = (('2022-09-05', '熊'), ('2024-08-05', '牛'))
ARMS = ('blend', 'nam')
DIRS = ('diagnose_output', os.path.join('diagnose_output', 'iter_output'))

_RE = {
    'ret': re.compile(r'总收益率:\s*(-?[\d.]+)%'),
    'dd': re.compile(r'最大回撤:\s*(-?[\d.]+)%'),
    'sharpe': re.compile(r'夏普比率:\s*(-?[\d.]+)'),
    'beta': re.compile(r'beta\s*:\s*(-?[\d.]+)'),
}
_RE_BENCH = re.compile(r'全市场等权\(日频再平衡\)\s+(-?[\d.]+)%')


def _read(arm, seed, win):
    for d in DIRS:
        p = os.path.join(d, f'T095_bt_{arm}_s{seed}_{win}.log')
        if os.path.isfile(p):
            with open(p, encoding='utf-8', errors='replace') as f:
                return f.read()
    return None


def _parse(txt):
    out = {}
    for k, rx in _RE.items():
        m = rx.search(txt)
        out[k] = float(m.group(1)) if m else np.nan
    m = _RE_BENCH.search(txt)
    out['bench'] = float(m.group(1)) if m else np.nan
    out['excess'] = out['ret'] - out['bench']
    return out


def main():
    data = {}       # (arm, seed, win) -> dict
    for arm in ARMS:
        for s in SEEDS:
            for win, _ in WINDOWS:
                txt = _read(arm, s, win)
                if txt:
                    data[(arm, s, win)] = _parse(txt)

    if not data:
        print('没有 T095 回测日志')
        return

    for win, wname in WINDOWS:
        print(f'\n=== {wname}市窗口 {win} ===')
        print(f"{'seed':>6} {'blend超额':>10} {'nam超额':>10} {'Δ(blend−nam)':>14}"
              f" {'blend β':>9} {'nam β':>9}")
        deltas = []
        for s in SEEDS:
            b, n = data.get(('blend', s, win)), data.get(('nam', s, win))
            if not (b and n):
                print(f'{s:>6}  {"（缺）" if not b else ""}{"（缺 nam）" if not n else ""}')
                continue
            d = b['excess'] - n['excess']
            deltas.append(d)
            print(f'{s:>6} {b["excess"]:>9.2f}pp {n["excess"]:>9.2f}pp {d:>13.2f}pp'
                  f' {b["beta"]:>9.3f} {n["beta"]:>9.3f}')
        if len(deltas) >= 2:
            deltas = np.asarray(deltas)
            mde = 2.0 * deltas.std(ddof=1) / np.sqrt(len(deltas))
            print(f'  配对 Δ 均值 {deltas.mean():+.2f}pp  σ_paired {deltas.std(ddof=1):.2f}pp'
                  f'  → n={len(deltas)} 的 MDE ≈ {mde:.2f}pp')
            print(f'  {int((deltas > 0).sum())}/{len(deltas)} 种子 blend 胜')
            data[('_stat_', win)] = {'mean': deltas.mean(), 'mde': mde,
                                     'wins': int((deltas > 0).sum()), 'n': len(deltas)}

    # ── 判据裁决 ──────────────────────────────────────────────────────
    print(f"\n{'='*66}\n判决\n{'='*66}")
    stats = {w: data.get(('_stat_', w)) for w, _ in WINDOWS}
    if any(v is None for v in stats.values()):
        print('两个窗口的数据不齐，无法裁决。')
        return

    # 门 1：无 regime 债 —— 任一窗口输超过该窗 MDE 即出局
    debts = []
    for win, wname in WINDOWS:
        st = stats[win]
        if st['mean'] < -st['mde']:
            debts.append(f'{wname}市输 {abs(st["mean"]):.2f}pp（> MDE {st["mde"]:.2f}pp）')
    if debts:
        print(f'**门1 不通过**：blend 在 {"、".join(debts)}')
        print('  → 一窗大赢一窗大输是风格彩票。按判据 blend 出局，生产维持 NAM 候选。')
        return
    print('门1 通过：两个窗口都没有超出噪声的落后（无 regime 债）')

    # 门 2：净增益
    tot = sum(stats[w]['mean'] for w, _ in WINDOWS)
    tot_mde = np.sqrt(sum(stats[w]['mde'] ** 2 for w, _ in WINDOWS))
    if tot > tot_mde:
        print(f'门2 通过：两窗合计 Δ {tot:+.2f}pp > 合并 MDE {tot_mde:.2f}pp')
        print(f'\n**判决：混合（xgb+lgb 等权）进生产。**')
    else:
        print(f'门2 不通过：两窗合计 Δ {tot:+.2f}pp ≤ 合并 MDE {tot_mde:.2f}pp')
        print('  → 无净增益，进入门3（平局判据）')
        bb = np.mean([data[('blend', s, w)]['beta'] for s in SEEDS
                      for w, _ in WINDOWS if ('blend', s, w) in data])
        nb = np.mean([data[('nam', s, w)]['beta'] for s in SEEDS
                      for w, _ in WINDOWS if ('nam', s, w) in data])
        bd = np.mean([data[('blend', s, w)]['dd'] for s in SEEDS
                      for w, _ in WINDOWS if ('blend', s, w) in data])
        nd = np.mean([data[('nam', s, w)]['dd'] for s in SEEDS
                      for w, _ in WINDOWS if ('nam', s, w) in data])
        print(f'  平均 β：blend {bb:.3f} vs nam {nb:.3f}（越接近 1 越好）')
        print(f'  平均最大回撤：blend {bd:.2f}% vs nam {nd:.2f}%（越浅越好）')
        score_b = abs(bb - 1) - abs(nb - 1)
        winner = 'NAM' if (score_b > 0 and bd < nd) else \
                 '混合' if (score_b < 0 and bd > nd) else \
                 ('混合' if score_b < 0 else 'NAM')
        print(f'\n**判决：{winner} 进生产（平局判据）。**')


if __name__ == '__main__':
    main()
