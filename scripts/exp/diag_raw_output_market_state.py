#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T137（零训练）—— 模型**原始输出**对市场状态有没有区分度？

来源：用户提问「回测里 min_confidence 就是控制这个的，也许有用呢？」

先把 min_confidence 的口径钉死（见 ml_factor_strategy.py:405-427）：
    confidence = rankdata(probs)/n*100     ← **当日截面分位**
所以 `min_confidence=θ` 每天恒好放行 (100-θ)% 只股票，**与市场状态无关**，
结构上不可能产生「今天一只都不买」。它是固定比例筛子，不是择时开关。
本脚本第 ① 段用实测把这条钉死（而不是只靠读代码）。

但用户的**底层问题**没被这条否掉，而且没测过：
    分位化之前的**原始分数 probs**，其**逐日水平**（均值/最大值/头部均值/离散度）
    是否携带市场状态信息？若携带，就可以改用**原始分数阈值**做择时。

先验（为什么值得测，也为什么可能是零）：
  · ListNet 损失对逐日整体平移**严格不变** ⇒ 逐日水平**从未收到过梯度**。
    它不是被训成 0，而是**完全无约束**的副产品。
  · 输入里大部分列是逐日截面 rank（每天分布几乎相同 ⇒ 水平应近似常数），
    但 skip-rank 连续列走的是**全局** robust-sigmoid（norm_stats 是全局的），
    指数相对特征（T115 起的 224 列）尤其可能带市场水平 ⇒ **不能先验判零**。

取数：直接复用 T133 的族输出缓存（695 MB），**零 GPU、零重训**。
  disable_gate 下 score = group_sums.sum(-1) + bias（nam_gate_model.py:218-221），
  bias 是全局常数，对**逐日水平的相对比较**没有影响 ⇒ sum(1) 就是原始分数。

预注册判据（三条，任一过 ⇒ 值得做真正的择时实验；全不过 ⇒ 关轴）：
  (1) 相关性：逐日统计量 vs 未来 7 日市场收益，holdout 上 Spearman 的
      Newey-West（lag=7，重叠收益必须校正）|t| ≥ 2，且 ≥3/4 存档同号。
  (2) 方向：「统计量高于中位数 ⇒ 猜涨」的准确率 ≥ **基准涨日率 + 3pp**。
      ⚠ 必须对基准涨日率判，不能对 50% 判 —— 等权市场上漂，
        绝对阈值会把没有方向技能的模型判成通过（T135 我犯过这个错）。
  (3) 经济意义：按统计量分最低五分位的那些日子，未来 7 日市场收益
      至少比全样本均值差 0.5pp，且 ≥3/4 存档同号，
      **且移动块自助（block=10）p < 0.05**。
      ⚠ 这最后半句是**首版漏掉、事后补上**的：首版只看幅度和 4/4 同号，
        于是 `std` 那一格被判成「过门」。但 (a) 4 个存档看同一批 161 天、
        同一族配方，分数高度相关，「4/4」不是 4 个独立证据；
        (b) 7 日重叠收益自相关 0.86，低五分位的 33 天只是 9 段行情。
        补上自助检验后 p=0.405，且效应在选型段**符号翻转**（+0.63% vs −1.42%）
        ⇒ 该格实为噪声。判据收紧的方向对原猜想不利，故非事后挪门。
        完整自查见 scripts/exp/diag_score_std_selfcheck.py。

对照（防止「target 本身是垃圾」的假阴性）：
  报告 mkt[i] 与 mkt[i-1] 的自相关 —— 7 日重叠收益必须强正，若不是则取数有问题。
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def newey_west_t(x, y, lag):
    """OLS y = a + b·x 的斜率 Newey-West t 值（重叠样本必须用，否则 t 虚高 ~sqrt(lag)）。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 20:
        return float('nan'), float('nan')
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1.0)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(max(V[1, 1], 1e-30)))
    return float(beta[1]), float(beta[1] / se)


STATS = ('mean', 'max', 'top20', 'std', 'p90_p50')


def block_bootstrap_p(v, y, q=0.2, n_boot=2000, block=10, seed=0):
    """低分位效应的移动块自助 p 值（单尾：效应为负）。

    成对块重采样保留时间依赖 —— 7 日重叠收益自相关 0.86，
    普通 t / 简单置换会把「几段行情」当成几十个独立样本。
    """
    rng = np.random.default_rng(seed)
    n = len(v)
    obs = float(y[v <= np.quantile(v, q)].mean() - y.mean())
    nb = int(np.ceil(n / block))
    pool = np.arange(0, n - block + 1)
    cnt = 0
    for _ in range(n_boot):
        st = rng.choice(pool, size=nb)
        i2 = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        vb, yb = v[i2], y[i2]
        if float(yb[vb <= np.quantile(vb, q)].mean() - yb.mean()) <= obs:
            cnt += 1
    return obs, float(cnt / n_boot)


def day_stats(sc):
    """一天的原始分数向量 → 5 个逐日水平统计量。"""
    kk = min(20, len(sc))
    top = np.partition(sc, -kk)[-kk:]
    return {'mean': float(sc.mean()),
            'max': float(sc.max()),
            'top20': float(top.mean()),
            'std': float(sc.std()),
            'p90_p50': float(np.percentile(sc, 90) - np.percentile(sc, 50))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-npz', default='diagnose_output/T133_groupsums.npz')
    # T133 存档顺序：for YS in 1 2 4 8 → for SD in 42 11
    ap.add_argument('--model-idx', default='0,1,2,3',
                    help='用 npz 里的哪几个存档（默认 y1_s42,y1_s11,y2_s42,y2_s11；'
                         'y4/y8 已被 T133 判负，水平分析没有生产意义）')
    ap.add_argument('--model-names', default='y1_s42,y1_s11,y2_s42,y2_s11')
    ap.add_argument('--select-holdout', type=float, default=0.4)
    ap.add_argument('--min-stocks', type=int, default=30)
    ap.add_argument('--horizon', type=int, default=7, help='标签前视天数 = NW lag')
    ap.add_argument('--conf-thresholds', default='50,90,95,99',
                    help='① 段要实测的 min_confidence 分位阈值')
    ap.add_argument('--out', default='diagnose_output/T137_raw_output_market_state.json')
    a = ap.parse_args()

    idx = [int(v) for v in a.model_idx.split(',') if v.strip() != '']
    names = [v.strip() for v in a.model_names.split(',') if v.strip()]
    assert len(idx) == len(names), '--model-idx 与 --model-names 长度必须一致'

    print(f'读缓存 {a.cache_npz} ...')
    z = np.load(a.cache_npz, allow_pickle=False)
    ret_va = z['ret_va']
    va_days = [(int(s), int(e)) for s, e in z['va_days']]
    days = [(s, e) for s, e in va_days if e - s >= a.min_stocks]
    n_sel = int(round(len(days) * (1.0 - a.select_holdout)))
    n_hold = len(days) - n_sel
    print(f'  验证 {len(days)} 日，holdout（后 {a.select_holdout:.0%}）= {n_hold} 日')

    S = {}
    for j, nm in zip(idx, names):
        S[nm] = z[f'S_va_{j}'].astype(np.float32)
        print(f'  {nm}: S_va_{j} {S[nm].shape}')

    # 逐日市场收益（截面等权未来 7 日收益）
    mkt = np.array([float(np.asarray(ret_va[s:e], float).mean()) for s, e in days])

    # ---------- 对照：target 自相关（7 日重叠必须强正） ----------
    ac1 = float(np.corrcoef(mkt[1:], mkt[:-1])[0, 1])
    print(f'\n[对照] mkt 一阶自相关 = {ac1:+.4f}'
          f'   {"✓ 重叠结构正常" if ac1 > 0.5 else "⚠ 异常，取数存疑"}')

    base_up = float((mkt[n_sel:] > 0).mean())
    print(f'[基准] holdout 涨日率 = {base_up:.1%}（方向门必须赢它 +3pp，不是赢 50%）')

    # ---------- ① min_confidence 的结构性实测 ----------
    print(f'\n===== ① min_confidence 是分位阈值：每日放行比例恒定 =====')
    ref = names[0]
    ths = [float(v) for v in a.conf_thresholds.split(',')]
    pass_frac = {t: [] for t in ths}
    pass_cnt = {t: [] for t in ths}
    for i, (s, e) in enumerate(days):
        sc = S[ref][s:e].astype(np.float64).sum(1)
        conf = rankdata(sc, method='average') / len(sc) * 100.0
        for t in ths:
            m = conf >= t
            pass_frac[t].append(float(m.mean()))
            pass_cnt[t].append(int(m.sum()))
    print(f'{"阈值θ":>8s} {"日均放行比例":>13s} {"比例标准差":>11s} '
          f'{"最少放行只数":>13s} {"零放行日数":>11s}')
    conf_res = {}
    for t in ths:
        f = np.array(pass_frac[t])
        c = np.array(pass_cnt[t])
        conf_res[str(t)] = {'frac_mean': float(f.mean()), 'frac_std': float(f.std()),
                            'min_count': int(c.min()), 'zero_days': int((c == 0).sum())}
        print(f'{t:8.0f} {f.mean():13.4%} {f.std():11.2e} {int(c.min()):13d} '
              f'{int((c == 0).sum()):11d}')
    print('⇒ 放行比例逐日恒为 (100-θ)%（标准差是并列值造成的数值噪声），')
    print('  **零放行日数恒为 0** ⇒ min_confidence 结构上无法产生空仓日，不是择时开关。')

    # ---------- ② 原始输出逐日水平 vs 未来市场收益 ----------
    print(f'\n===== ② 原始分数逐日水平 → 未来 {a.horizon} 日市场收益（holdout {n_hold} 日）=====')
    lev = {nm: {k: np.full(len(days), np.nan) for k in STATS} for nm in names}
    for i, (s, e) in enumerate(days):
        for nm in names:
            st = day_stats(S[nm][s:e].astype(np.float64).sum(1))
            for k in STATS:
                lev[nm][k][i] = st[k]

    sl = slice(n_sel, None)
    mh = mkt[sl]
    res = {}
    print(f'{"存档":10s} {"统计量":>9s} {"Spearman":>10s} {"NW t":>8s} '
          f'{"方向准确":>9s} {"Δ基准":>8s} {"低五分位 mkt":>13s} {"Δ全样本":>9s} {"块自助p":>9s}')
    for nm in names:
        res[nm] = {}
        for k in STATS:
            v = lev[nm][k][sl]
            rho = float(spearmanr(v, mh).correlation)
            _, t = newey_west_t(rankdata(v), rankdata(mh), a.horizon)
            pred_up = v > np.median(v)
            acc = float((pred_up == (mh > 0)).mean())
            q = np.quantile(v, 0.2)
            low = mh[v <= q]
            _, boot_p = block_bootstrap_p(v, mh)
            res[nm][k] = {'spearman': rho, 'nw_t': t, 'acc': acc,
                          'acc_minus_base': acc - base_up,
                          'low_q_mkt': float(low.mean()),
                          'low_q_minus_all': float(low.mean() - mh.mean()),
                          'low_q_boot_p': boot_p,
                          'n_low': int(len(low))}
            r = res[nm][k]
            print(f'{nm:10s} {k:>9s} {rho:+10.4f} {t:+8.2f} {acc:9.1%} '
                  f'{r["acc_minus_base"]:+8.1%} {low.mean():+13.4%} '
                  f'{r["low_q_minus_all"]:+9.4%} {boot_p:9.3f}')

    # ---------- ③ 预注册判据 ----------
    print(f'\n===== ③ 预注册判据 =====')
    verdict = {}
    for k in STATS:
        ts = [res[nm][k]['nw_t'] for nm in names]
        rhos = [res[nm][k]['spearman'] for nm in names]
        accs = [res[nm][k]['acc_minus_base'] for nm in names]
        lows = [res[nm][k]['low_q_minus_all'] for nm in names]
        ps = [res[nm][k]['low_q_boot_p'] for nm in names]
        # (1) |t|≥2 且 ≥3/4 同号
        sgn = np.sign(rhos)
        agree = max(int((sgn > 0).sum()), int((sgn < 0).sum()))
        c1 = bool(sum(abs(t) >= 2 for t in ts) >= 3 and agree >= 3)
        # (2) 方向赢基准 +3pp（≥3/4）
        c2 = bool(sum(d >= 0.03 for d in accs) >= 3)
        # (3) 低五分位差 ≥0.5pp、≥3/4 同号，**且**块自助 p<0.05（≥3/4）
        c3 = bool(sum(d <= -0.005 for d in lows) >= 3
                  and sum(p < 0.05 for p in ps) >= 3)
        verdict[k] = {'c1_corr': c1, 'c2_direction': c2, 'c3_economic': c3,
                      'any': bool(c1 or c2 or c3),
                      't_values': ts, 'spearman': rhos,
                      'acc_minus_base': accs, 'low_q_minus_all': lows,
                      'low_q_boot_p': ps}
        print(f'{k:>9s}: (1)相关 {"过" if c1 else "不过"}  '
              f'(2)方向 {"过" if c2 else "不过"}  '
              f'(3)经济 {"过" if c3 else "不过"}  '
              f'⇒ {"值得深挖" if verdict[k]["any"] else "无"}')

    any_pass = any(v['any'] for v in verdict.values())
    print(f'\n★ 总判定：{"至少一条统计量过门 ⇒ 原始输出确实带市场状态信息，值得做真正的择时实验" if any_pass else "五个统计量全不过门 ⇒ 原始输出的逐日水平不携带可用的市场状态信息，本轴关闭"}')
    if not any_pass:
        print('  与 T135 一致：那次测的是 regime 向量 → 市场方向（不成立），')
        print('  这次测的是模型自己的原始输出水平 → 市场方向（也不成立）。')
        print('  两者独立地否掉了「熊市空仓/牛市梭哈」的先决条件。')

    out = {'window': {'n_days': len(days), 'n_select': n_sel, 'n_holdout': n_hold,
                      'base_up_rate': base_up, 'mkt_autocorr1': ac1,
                      'horizon': a.horizon},
           'min_confidence_structure': conf_res,
           'levels': res, 'verdict': verdict, 'any_pass': any_pass}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已写 {a.out}')


if __name__ == '__main__':
    main()
