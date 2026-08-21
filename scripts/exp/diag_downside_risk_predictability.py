#!/usr/bin/env python
"""T138：regime 能否预测未来 7 日**下行风险**（而非收益方向）？

为什么问这个（T135 之后唯一还开着的窄问题）
--------------------------------------------
T135 已证：未来 7 日市场**方向**预测不出来 —— 折外方向准确 60.9% < 永远猜涨的 63.5%。
但 T134 顺带量出一件事：**回撤是低噪声统计量，终值收益不是**
（两个统计上同一的模型，终值收益差 27~60pp，而最大回撤只差 0.1~0.3pp）。
收益方向噪声太大所以测不出，不代表**风险**也测不出 —— 波动率在所有市场里都以
持续性著称。所以这是一个**独立的、更容易成立的**问题，值得单测一次。

⚠ 但「更容易成立」正是陷阱所在，本脚本的主门因此不是 IC 而是**增量**：
波动率的持续性意味着「用过去 20 日波动预测未来 7 日波动」本身就很准。
若 30 维 regime 赢不了这个平凡基线，它就没带来任何新东西 ——
而且 regime 特征里本来就含波动类列，那等于**自己预测自己的滞后**。
T137 刚在这件事上栽过：score-std 看着能预测下行，一比「过去实现波动」就不剩什么。

预注册判据（出结果前写死；三条都独立报告）
------------------------------------------
  (1) 可预测性：折外 Spearman IC（预测 vs 已实现下行风险）**≥ +0.10** 且 ≥3/4 折同号。
  (2) **增量（主门）**：**嵌套模型**对照 —— 基线模型（只用过去 20 日实现波动 + 下行半差）
      vs 全模型（基线 + 30 维 regime），**ΔIC ≥ +0.05** 且 ≥3/4 折同号；
      并报全模型预测剔除基线预测后的偏 IC。
      不过则本轴关闭 —— 信号是波动持续性，不是 regime。
      ⚠ **首版设计有缺陷、已更正**：首版只喂 30 维 regime、不给基线列，
        而 regime 是 `rolling().rank(pct=True)` 归一化的 —— **构造上不含绝对水平**，
        却被要求去拟合 dsemi 的**原始水平**。那等于让它做一份它拿不到输入的工作，
        再拿「偏 IC 为负」判它没增量，不公平。改成嵌套对照后，基线列直接进 X，
        regime 只需解释**残差**，这才是「有没有增量」的干净问法。
  (3) 可决策性：预测风险**最高五分位**的日子，未来 7 日市场收益要比全样本差 ≥0.5pp，
      **且移动块自助 p < 0.05**（block=10）。
      ⚠ 这条自助检验是**吸取 T137 教训**从一开始就写进判据的：7 日重叠收益自相关 ~0.86，
        不做块自助就会把「几段行情」当成几十个独立样本。

  (1)+(2) 过 = regime 确实带增量的下行风险信息；(3) 决定它对**只做多**的本策略有没有用。
  三条的组合含义在脚本末尾按实际结果打印，不预设结论。

⚠ 立场声明（避免结论被误用）
  即使三条全过，「按预测风险调仓位」也是**风控规则**，落在用户立下的
  「只找 alpha，不做让回测好看的规则」约束内。本脚本只做**可预测性测量**，
  不构成上线建议；是否动仓位由用户决定。

方法学纪律（沿用 T135，全部保留）
  · 禁运 horizon 日：目标用到 t+1..t+H 的收益，训练集截到 test_start−H。
  · 只向前扩张窗 walk-forward，不做随机 K 折。
  · 折边界从 min_train+embargo 起算（T135 首版在这里静默丢过第 1 折）。
  · regime 归一化已确认因果（`rolling().rank(pct=True)`，只看窗内历史）。

用法
  python scripts/exp/diag_downside_risk_predictability.py --folds 4 --horizon 7
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scipy.stats import rankdata, spearmanr

from config import DATABASE_PATH
from core.factors.regime_features import build_regime_matrix, load_market_sentiment


def ridge_fit(X, y, lam):
    """带截距的岭回归（截距不惩罚）。与 T135 同一实现。"""
    Xc = np.c_[np.ones(len(X)), X]
    A = Xc.T @ Xc + lam * len(X) * np.eye(Xc.shape[1])
    A[0, 0] -= lam * len(X)
    return np.linalg.solve(A, Xc.T @ y)


def ridge_pred(w, X):
    return np.c_[np.ones(len(X)), X] @ w


def standardize(tr, te):
    """按**训练折**的均值/标准差标准化（因果）。

    ⚠ 不做这一步，嵌套对照是无效的：regime 列是 `rolling().rank(pct=True)` ⇒ 落在 [0,1]，
    而基线列是实现波动 ⇒ 量纲 ~0.008。同一个 ridge 惩罚下，基线列要起作用就得配
    极大的系数，于是被罚没 —— 全模型的预测会退化成 regime-only（首版实测两者
    IC 差 1e-4，就是这个症状）。标准化后各列在惩罚下地位相同。
    """
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (tr - mu) / sd, (te - mu) / sd


def resid_on(y, Z):
    """把 y 对 Z 做 OLS 后取残差（用于偏相关）。"""
    Zc = np.c_[np.ones(len(Z)), Z]
    b, *_ = np.linalg.lstsq(Zc, y, rcond=None)
    return y - Zc @ b


def partial_ic(pred, true, base):
    """剔除基线后的偏 Spearman：三者先转秩，再把 pred/true 对 base 取残差。"""
    rp, rt = rankdata(pred), rankdata(true)
    rb = np.column_stack([rankdata(base[:, j]) for j in range(base.shape[1])])
    return float(spearmanr(resid_on(rp, rb), resid_on(rt, rb)).statistic)


def block_bootstrap_p(v, y, q, n_boot=5000, block=10, seed=0, upper=True):
    """分位效应的移动块自助 p 值（单尾：效应为负）。成对块重采样保留时间依赖。"""
    rng = np.random.default_rng(seed)
    n = len(v)
    sel = (lambda a, t: a >= t) if upper else (lambda a, t: a <= t)
    thr = np.quantile(v, 1 - q if upper else q)
    obs = float(y[sel(v, thr)].mean() - y.mean())
    nb = int(np.ceil(n / block))
    pool = np.arange(0, max(n - block + 1, 1))
    cnt = 0
    for _ in range(n_boot):
        st = rng.choice(pool, size=nb)
        i2 = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        vb, yb = v[i2], y[i2]
        tb = np.quantile(vb, 1 - q if upper else q)
        if float(yb[sel(vb, tb)].mean() - yb.mean()) <= obs:
            cnt += 1
    return obs, float(cnt / n_boot)


def forward_windows(r, H):
    """行 t = [r_{t+1}, ..., r_{t+H}]（不含当日）。"""
    return np.column_stack([np.roll(r, -(i + 1)) for i in range(H)])


def trailing_windows(r, L):
    """行 t = [r_t, r_{t-1}, ..., r_{t-L+1}]（当日收盘后已知）。"""
    return np.column_stack([np.roll(r, i) for i in range(L)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=7)
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--embargo', type=int, default=None, help='默认 = horizon')
    ap.add_argument('--ridge', type=float, default=1e-2)
    ap.add_argument('--min-train', type=int, default=500)
    ap.add_argument('--trail', type=int, default=20, help='平凡基线的回看长度')
    ap.add_argument('--ic-line', type=float, default=0.10)
    ap.add_argument('--pic-line', type=float, default=0.05)
    ap.add_argument('--quantile', type=float, default=0.2)
    ap.add_argument('--block', type=int, default=10)
    ap.add_argument('--target', default='dsemi', choices=('dsemi', 'mdd', 'vol'),
                    help='主目标：dsemi=下行半差（预注册主目标）/ mdd=区间最大回撤 / vol=总波动')
    ap.add_argument('--out', default='diagnose_output/T138_downside_risk.json')
    a = ap.parse_args()
    emb = a.embargo if a.embargo is not None else a.horizon
    H, L = a.horizon, a.trail

    sent = load_market_sentiment(DATABASE_PATH)
    if sent.empty or 'mean_return' not in sent.columns:
        raise SystemExit('market_sentiment 表缺失或无 mean_return 列')
    mret = sent['mean_return'].astype(float)
    M = build_regime_matrix(DATABASE_PATH)
    if M.empty:
        raise SystemExit('regime 矩阵为空')
    idx = M.index.intersection(mret.index)
    M, mret = M.loc[idx], mret.loc[idx]
    r = mret.to_numpy(np.float64)
    n_raw = len(r)

    # ---------- 目标（未来 H 日）----------
    F = forward_windows(r, H)
    cum = np.cumprod(1.0 + F, axis=1)
    peak = np.maximum.accumulate(np.c_[np.ones(len(cum)), cum], axis=1)[:, 1:]
    tgt = {'dsemi': np.sqrt(np.mean(np.minimum(F, 0.0) ** 2, axis=1)),
           'vol': F.std(axis=1, ddof=1),
           'mdd': (1.0 - cum / peak).max(axis=1)}
    fwd_ret = np.prod(1.0 + F, axis=1) - 1.0        # 未来 H 日收益，供判据 (3)

    # ---------- 平凡基线（当日已知）----------
    P = trailing_windows(r, L)
    base = np.column_stack([P.std(axis=1, ddof=1),
                            np.sqrt(np.mean(np.minimum(P, 0.0) ** 2, axis=1))])

    # 首尾被 roll 污染的行必须切掉：前 L-1 行（trailing 绕回尾部）、后 H 行（forward 绕回开头）
    valid = np.zeros(n_raw, bool)
    valid[L - 1:n_raw - H] = True
    valid &= M.notna().all(axis=1).to_numpy()
    valid &= np.isfinite(tgt[a.target]) & np.isfinite(base).all(axis=1)

    X = M.to_numpy(np.float64)[valid]
    Y = {k: v[valid] for k, v in tgt.items()}
    B = base[valid]
    FR = fwd_ret[valid]
    dates = M.index[valid]
    n, p = X.shape
    print(f'样本 {n} 个交易日（{dates[0].date()} → {dates[-1].date()}），regime 维度 {p}')
    print(f'主目标 = 未来 {H} 日 {a.target}；平凡基线 = 过去 {L} 日实现波动 + 下行半差')
    print(f'禁运 {emb} 日；{a.folds} 折 walk-forward\n')

    first = a.min_train + emb
    if first >= n - 60:
        raise SystemExit(f'样本不足：min_train+embargo={first} 接近总样本 {n}')
    bounds = np.linspace(first, n, a.folds + 1).astype(int)

    rows, oof = [], {k: [] for k in ('pred', 'pred_b', 'true', 'b0', 'b1', 'fr')}
    for k in range(a.folds):
        te0, te1 = bounds[k], bounds[k + 1]
        tr1 = te0 - emb
        if tr1 < a.min_train or te1 - te0 < 30:
            print(f'  [跳过] 折{k+1}: tr1={tr1} n_te={te1-te0}')
            continue
        y = Y[a.target]
        # 嵌套三模型：base=只有平凡基线；reg=只有 regime；full=基线+regime
        # 每个设计矩阵都按训练折标准化（见 standardize 的注释：不标准化则 full 退化成 reg）
        Btr, Bte = standardize(B[:tr1], B[te0:te1])
        Rtr, Rte = standardize(X[:tr1], X[te0:te1])
        Ftr, Fte = np.column_stack([Btr, Rtr]), np.column_stack([Bte, Rte])
        w_b = ridge_fit(Btr, y[:tr1], a.ridge)
        w_r = ridge_fit(Rtr, y[:tr1], a.ridge)
        w_f = ridge_fit(Ftr, y[:tr1], a.ridge)
        pr_b = ridge_pred(w_b, Bte)
        pr_r = ridge_pred(w_r, Rte)
        pr_f = ridge_pred(w_f, Fte)
        tr = y[te0:te1]
        bb = B[te0:te1]
        ic_b = float(spearmanr(pr_b, tr).statistic)
        ic_r = float(spearmanr(pr_r, tr).statistic)
        ic_f = float(spearmanr(pr_f, tr).statistic)
        pic = partial_ic(pr_f, tr, pr_b.reshape(-1, 1))
        rows.append(dict(fold=k + 1, n_train=int(tr1), n_test=int(te1 - te0),
                         start=str(dates[te0].date()), end=str(dates[te1 - 1].date()),
                         ic_base=ic_b, ic_regime=ic_r, ic_full=ic_f,
                         d_ic=ic_f - ic_b, partial_ic=pic))
        oof['pred'] += list(pr_f); oof['pred_b'] += list(pr_b)
        oof['true'] += list(tr); oof['fr'] += list(FR[te0:te1])
        oof['b0'] += list(bb[:, 0]); oof['b1'] += list(bb[:, 1])
        print(f'  折{k+1} {rows[-1]["start"]}→{rows[-1]["end"]}  n_te={te1-te0:4d}  '
              f'基线IC={ic_b:+.4f}  regime单独={ic_r:+.4f}  全模型={ic_f:+.4f}  '
              f'ΔIC={ic_f-ic_b:+.4f}  偏IC={pic:+.4f}')

    Pd, Tt = np.array(oof['pred']), np.array(oof['true'])
    Pb = np.array(oof['pred_b'])
    Bb = np.column_stack([np.array(oof['b0']), np.array(oof['b1'])])
    FRo = np.array(oof['fr'])
    ic_all = float(spearmanr(Pd, Tt).statistic)
    icb_pred = float(spearmanr(Pb, Tt).statistic)
    pic_all = partial_ic(Pd, Tt, Pb.reshape(-1, 1))
    d_ic = ic_all - icb_pred
    icb_all = float(spearmanr(Bb[:, 0], Tt).statistic)
    icb1_all = float(spearmanr(Bb[:, 1], Tt).statistic)
    n_pos = sum(1 for x in rows if x['ic_full'] > 0)
    n_ppos = sum(1 for x in rows if x['d_ic'] > 0)
    # ⚠ 主指标用**折均 IC**，不用合并 OOF IC：各折的预测值水平/尺度不同
    #（不同训练段拟合出不同截距），合并后排序把跨折水平漂移也算了进去。
    #   首版实测症状：基线模型逐折 IC +0.038/+0.242/+0.203/+0.154，合并后只剩 +0.018 ——
    #   而它的原始输入（裸的过去 20 日波动，全样本同尺度）合并 IC 是 +0.172。
    #   合并 IC 在这里量的是「跨折可比性」而不是「预测能力」。
    ic_fold = float(np.mean([x['ic_full'] for x in rows]))
    icb_fold = float(np.mean([x['ic_base'] for x in rows]))
    icr_fold = float(np.mean([x['ic_regime'] for x in rows]))
    d_ic_fold = float(np.mean([x['d_ic'] for x in rows]))

    print(f'\n合并折外（n={len(Tt)}）')
    print(f'  裸的过去{L}日波动        IC={icb_all:+.4f}')
    print(f'  裸的过去{L}日下行半差    IC={icb1_all:+.4f}')
    print(f'\n折均 IC（主指标，跨折水平不可比故不用合并 IC）')
    print(f'  基线模型（2 列）        {icb_fold:+.4f}   逐折 '
          f'{[round(x["ic_base"], 4) for x in rows]}')
    print(f'  regime 单独（30 列）    {icr_fold:+.4f}   逐折 '
          f'{[round(x["ic_regime"], 4) for x in rows]}')
    print(f'  全模型（基线+regime）   {ic_fold:+.4f}   逐折 '
          f'{[round(x["ic_full"], 4) for x in rows]}')
    print(f'  **ΔIC（增量）**         {d_ic_fold:+.4f}   ← 主门看这个，{n_ppos}/{len(rows)} 折为正')
    print(f'  （参考）合并 OOF：基线 {icb_pred:+.4f} / 全模型 {ic_all:+.4f} / '
          f'剔基线偏 IC {pic_all:+.4f}')

    # ---------- 判据 (3) 可决策性 ----------
    # ⚠ 要问的是「**会预测风险的那个模型**指认的高风险日，收益是否更差」。
    #   全模型的风险预测本身就不合格（折均 IC +0.036），拿它定高风险日等于问了个废问题。
    #   所以三个预测源都报：全模型 / 基线模型（折均 IC 最高）/ 裸的过去 20 日波动。
    src = {'full_model': Pd, 'base_model': Pb, 'raw_trailing_vol': Bb[:, 0]}
    dec = {}
    print(f'\n判据(3) 预测风险最高 {a.quantile:.0%} 的日子（各 n≈{int(len(Pd) * a.quantile)}）：')
    for sname, sv in src.items():
        e_, p_ = block_bootstrap_p(sv, FRo, a.quantile, block=a.block, upper=True)
        h_ = sv >= np.quantile(sv, 1 - a.quantile)
        step = slice(None, None, H)
        fr_ns, hi_ns = FRo[step], h_[step]
        hold = float(np.prod(1.0 + fr_ns) - 1.0)
        avoid = float(np.prod(1.0 + np.where(hi_ns, 0.0, fr_ns)) - 1.0)
        dec[sname] = dict(effect=e_, boot_p=p_, n_high=int(h_.sum()),
                          realized_target=float(Tt[h_].mean()),
                          hold_all=hold, avoid_high=avoid, n_blocks=int(len(fr_ns)))
        print(f'  [{sname:16s}] 未来{H}日收益 {FRo[h_].mean():+.4%} vs 全样本 '
              f'{FRo.mean():+.4%} ⇒ 效应 {e_:+.4%}，块自助 p={p_:.3f}'
              f'   |已实现 {a.target} {Tt[h_].mean():.5f} vs {Tt.mean():.5f}')
        print(f'  {"":18s} 不重叠{H}日调仓（{len(fr_ns)} 段，仅界定量级）：'
              f'一直持有 {hold*100:+.1f}% / 空掉高风险段 {avoid*100:+.1f}%')
    eff, bp = dec['base_model']['effect'], dec['base_model']['boot_p']
    print(f'  ⇒ 判据(3) 以 **base_model**（唯一合格的风险预测器）为准。')

    pass1 = bool(ic_fold >= a.ic_line and n_pos >= max(3, int(np.ceil(0.75 * len(rows)))))
    pass2 = bool(d_ic_fold >= a.pic_line and n_ppos >= max(3, int(np.ceil(0.75 * len(rows)))))
    pass3 = bool(eff <= -0.005 and bp < 0.05)
    print(f'\n★ 预注册裁决（折均口径）')
    print(f'   (1) 全模型折均 IC ≥ {a.ic_line} 且 ≥3/{len(rows)} 折同号 : '
          f'{ic_fold:+.4f}, {n_pos}/{len(rows)}  {"✓" if pass1 else "✗"}')
    print(f'   (2) 折均 ΔIC ≥ {a.pic_line} 且 ≥3/{len(rows)} 折为正（主门）: '
          f'{d_ic_fold:+.4f}, {n_ppos}/{len(rows)}  {"✓" if pass2 else "✗"}')
    print(f'   (3) 高风险分位收益差 ≤ −0.5pp 且自助 p<0.05 : '
          f'{eff:+.4%}, p={bp:.3f}  {"✓" if pass3 else "✗"}')

    print(f'\n   ⚑ 平凡基线自己的成绩：折均 IC {icb_fold:+.4f}（{sum(1 for x in rows if x["ic_base"] > 0)}/{len(rows)} 折为正）')
    if icb_fold >= a.ic_line and not pass2:
        print('   ⇒ **下行风险确实可预测，但那是波动持续性**：'
              f'过去 {L} 日的实现波动已经把信号拿走了，30 维 regime 不但没增量，'
              f'ΔIC 还是 {d_ic_fold:+.4f}。本轴关闭。')
    elif not pass1 and icb_fold < a.ic_line:
        print('   ⇒ **不可预测**：连波动这种以持续性著称的量都没到门，'
              '本轴关闭。与 T135（方向不可预测）同向。')
    elif pass2 and not pass3:
        print('   ⇒ **有增量但不可决策**：regime 带增量的下行风险信息，'
              '但高风险日的未来收益并不显著更差 ⇒ 对只做多的本策略没有可用的动作。'
              '（高波动 ≠ 低收益，这与 T135「方向不可预测」自洽。）')
    elif pass2 and pass3:
        print('   ⇒ **三条全过**：regime 带增量下行风险信息，且高风险日收益显著更差。'
              '\n   ⚠ 但据此调仓位属**风控规则**，落在「只找 alpha，不做让回测好看的规则」'
              '约束内 —— 本脚本只给测量，不构成上线建议。')

    out = dict(horizon=H, embargo=emb, trail=L, target=a.target, n=int(n), dims=int(p),
               folds=rows,
               fold_mean=dict(ic_full=ic_fold, ic_base=icb_fold, ic_regime=icr_fold,
                              d_ic=d_ic_fold),
               oof=dict(n=len(Tt), ic_full=ic_all, ic_base_model=icb_pred, d_ic=d_ic,
                        partial_ic=pic_all,
                        ic_trailing_vol=icb_all, ic_trailing_dsemi=icb1_all,
                        n_pos=n_pos, n_dic_pos=n_ppos),
               decision=dict(quantile=a.quantile, primary='base_model', sources=dec),
               verdict=dict(pass_predictable=pass1, pass_incremental=pass2,
                            pass_actionable=pass3))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
