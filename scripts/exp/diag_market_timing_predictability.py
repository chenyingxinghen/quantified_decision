#!/usr/bin/env python
"""T135：**市场择时的先决条件测试** —— 未来 7 日市场方向可预测吗？

为什么问这个
------------
当前模型**在构造上**不可能择时，三层都被去掉了市场水平信息：
  ① 标签 = 逐日**截面排名** ⇒ 每天的标签分布完全相同，与大盘涨跌无关
  ② 损失 = `-Σ tgt·log_softmax(score)` ⇒ **对逐日整体平移严格不变**；
     模型把某天所有分数减 10，损失分毫不动、梯度为零
  ③ 输出 = `rankdata(probs)/n*100` ⇒ 第一名恒为 ~100
所以「熊市空仓、牛市梭哈」缺的不是模型容量，是**训练目标里没有这个维度**。

要补这个维度，先决条件只有一条：**未来市场方向本身可预测**。
不可预测的话，无论怎么改标签/损失/仓位映射都没意义 —— 先花 10 分钟证伪掉，
比花几天改管线再发现学不出来划算。

T134 给了这条轴一个具体的价钱：高集中臂（y4/y8）的 α 在熊市 −23.7%、牛市 +18.7%，
窗间摆动 +33~+50pp 且 4/4 一致。那份超额**只有知道自己在哪个 regime 才领得到**。

已有的负面先验（但**都不是**直接测这个问题）
  · T130: `ifelse_oracle`（按**已实现**市场方向分支）头部超额 +0.0095 vs `global` +0.0042，
    而 `regime`（30 维线性、事前）−0.0010（1/4 种子）⇒ 开关挂在未来
  · 门控轴 T118/T119/T128 2×2 全负（[[nam-gate-collapses-to-status]]）
  ⚠ 但以上测的都是"今天该信哪个因子族"，**不是**"大盘会涨还是会跌"。
    后者从未直接测过。目标不同，不能直接外推，所以值得单独测一次。

预注册判据（出结果前写死）
--------------------------
  主判据：折外（walk-forward）预测值 vs 已实现未来 7 日市场收益的 **Spearman IC**
    可行门：**IC ≥ 0.10** 且 **≥3/4 折同号** 且 **方向准确率 ≥ 55%**
    低于此门 ⇒ 择时的先决条件不成立，整条思路关闭（不是"再调调看"）。
  0.10 / 55% 这条线的来历：今天 T134 实测出回测终值收益的噪声地板是 27~60pp
    （两个统计上同一的模型，同口径同窗）。比这更弱的择时信号，其收益改善
    在回测里**根本验证不了**，等于不可证伪。
  必比基线：**朴素动量**（过去 20 日市场收益）。30 维 regime 向量若赢不了
    "最近涨了就看多"，它就没带来任何东西。
  另报：**oracle**（完美预知）的价值上界，用来界定这条轴值多少钱。

方法学纪律
----------
  · **禁运 7 日**：第 j 日的目标用到 j+1..j+7 的收益，训练集必须截到 test_start−7，
    否则训练集见过测试期的收益。这正是 [[rolling-stats-of-forward-derived-quantities-need-embargo]]
    里把 persist 臂虚增 73% 的那个坑。
  · **只向前**：扩张窗 walk-forward，不做随机 K 折（时序数据随机折必然泄漏）。
  · regime 归一化本身已确认因果（`rolling().rank(pct=True)`，只看窗内历史）。

用法
  python scripts/exp/diag_market_timing_predictability.py --folds 4 --horizon 7
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

import pandas as pd
from scipy.stats import spearmanr

from config import DATABASE_PATH
from core.factors.regime_features import build_regime_matrix, load_market_sentiment


def ridge_fit(X, y, lam):
    """带截距的岭回归；X 已中心化时截距即 y 均值。"""
    Xc = np.c_[np.ones(len(X)), X]
    A = Xc.T @ Xc + lam * len(X) * np.eye(Xc.shape[1])
    A[0, 0] -= lam * len(X)                      # 不惩罚截距
    return np.linalg.solve(A, Xc.T @ y)


def ridge_pred(w, X):
    return np.c_[np.ones(len(X)), X] @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=7, help='预测未来 N 个交易日的市场收益')
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--embargo', type=int, default=None, help='默认 = horizon')
    ap.add_argument('--ridge', type=float, default=1e-2)
    ap.add_argument('--min-train', type=int, default=500)
    ap.add_argument('--ic-line', type=float, default=0.10)
    ap.add_argument('--acc-line', type=float, default=0.55)
    ap.add_argument('--out', default='diagnose_output/T135_market_timing.json')
    a = ap.parse_args()
    emb = a.embargo if a.embargo is not None else a.horizon

    sent = load_market_sentiment(DATABASE_PATH)
    if sent.empty or 'mean_return' not in sent.columns:
        raise SystemExit('market_sentiment 表缺失或无 mean_return 列')
    mret = sent['mean_return'].astype(float)

    M = build_regime_matrix(DATABASE_PATH)
    if M.empty:
        raise SystemExit('regime 矩阵为空')
    idx = M.index.intersection(mret.index)
    M = M.loc[idx]
    mret = mret.loc[idx]

    # 目标：未来 horizon 个交易日的累计市场收益（不含当日）
    fwd = (1.0 + mret).shift(-1).rolling(a.horizon).apply(np.prod, raw=True).shift(-(a.horizon - 1)) - 1.0
    ok = fwd.notna() & M.notna().all(axis=1)
    X = M[ok].to_numpy(np.float64)
    y = fwd[ok].to_numpy(np.float64)
    dates = M.index[ok]
    # 朴素动量基线：过去 20 日市场累计收益（当日可知）
    mom = ((1.0 + mret).rolling(20).apply(np.prod, raw=True) - 1.0)[ok].to_numpy(np.float64)

    n, p = X.shape
    print(f'样本 {n} 个交易日（{dates[0].date()} → {dates[-1].date()}），regime 维度 {p}')
    print(f'目标 = 未来 {a.horizon} 日市场累计收益；禁运 {emb} 日；{a.folds} 折 walk-forward\n')

    # 折边界从 min_train+embargo 起算，否则第 1 折的 tr1 会小于 min_train 被静默跳过
    #（首版就踩了：4 折只跑了 3 折，而"折数"又进了判据，等于判据被悄悄改小）
    first = a.min_train + emb
    if first >= n - 60:
        raise SystemExit(f'样本不足：min_train+embargo={first} 已接近总样本 {n}')
    bounds = np.linspace(first, n, a.folds + 1).astype(int)
    rows, oof_pred, oof_true, oof_mom = [], [], [], []
    for k in range(a.folds):
        te0, te1 = bounds[k], bounds[k + 1]
        tr1 = te0 - emb                      # 禁运：训练集不能碰到测试期目标用到的收益
        if tr1 < a.min_train or te1 - te0 < 30:
            print(f'  [跳过] 折{k+1}: tr1={tr1} n_te={te1-te0}（不满足 min_train/最小测试长度）')
            continue
        w = ridge_fit(X[:tr1], y[:tr1], a.ridge)
        pr = ridge_pred(w, X[te0:te1])
        tr = y[te0:te1]
        ic = spearmanr(pr, tr).statistic
        acc = float(((pr > 0) == (tr > 0)).mean())
        ic_m = spearmanr(mom[te0:te1], tr).statistic
        rows.append(dict(fold=k + 1, n_train=int(tr1), n_test=int(te1 - te0),
                         start=str(dates[te0].date()), end=str(dates[te1 - 1].date()),
                         ic=float(ic), acc=acc, ic_mom=float(ic_m),
                         base_up=float((tr > 0).mean())))
        oof_pred += list(pr); oof_true += list(tr); oof_mom += list(mom[te0:te1])
        print(f'  折{k+1} {rows[-1]["start"]}→{rows[-1]["end"]}  n_te={te1-te0:4d}  '
              f'IC={ic:+.4f}  方向准确={acc*100:5.1f}%  (基准涨日率 {rows[-1]["base_up"]*100:.1f}%)  '
              f'动量IC={ic_m:+.4f}')

    P, T, MO = np.array(oof_pred), np.array(oof_true), np.array(oof_mom)
    ic_all = float(spearmanr(P, T).statistic)
    acc_all = float(((P > 0) == (T > 0)).mean())
    ic_mom_all = float(spearmanr(MO, T).statistic)
    base_up = float((T > 0).mean())
    n_pos = sum(1 for r in rows if r['ic'] > 0)

    print(f'\n合并折外（n={len(T)}）')
    print(f'  regime 30 维  IC={ic_all:+.4f}   方向准确={acc_all*100:.1f}%')
    print(f'  朴素动量20d   IC={ic_mom_all:+.4f}')
    print(f'  基准涨日率    {base_up*100:.1f}%（方向准确率要跟它比，不是跟 50% 比）')

    # oracle：完美预知的价值上界 —— 只在预知为正的日子持有
    hold_all = float(np.prod(1.0 + T[::a.horizon]) - 1.0)
    hold_orc = float(np.prod(1.0 + np.maximum(T[::a.horizon], 0.0)) - 1.0)
    print(f'\n  价值上界（每 {a.horizon} 日不重叠调仓，仅供界定量级）:')
    print(f'    一直持有 {hold_all*100:+.1f}%   完美预知只在涨段持有 {hold_orc*100:+.1f}%')

    pass_ic = ic_all >= a.ic_line
    pass_sign = n_pos >= max(3, int(np.ceil(0.75 * len(rows))))
    # ⚠ 预注册时把方向门写成"绝对 55%"是**设计错误**：等权全市场是上漂的，
    #   基准涨日率就有 ~62%，"永远猜涨"白拿 62%。绝对阈值把一个没有方向技能的
    #   模型判成通过。改成"必须赢基准涨日率"——这个更正**收紧**了判据，不是放松。
    pass_acc = acc_all >= base_up + 0.03
    beats_mom = ic_all > ic_mom_all
    ok_all = pass_ic and pass_sign and pass_acc
    print(f'\n★ 预注册裁决')
    print(f'   IC ≥ {a.ic_line}                 : {ic_all:+.4f}  {"✓" if pass_ic else "✗"}')
    print(f'   ≥3/{len(rows)} 折同号             : {n_pos}/{len(rows)}  {"✓" if pass_sign else "✗"}')
    print(f'   方向准确 ≥ 基准涨日率+3pp  : {acc_all*100:.1f}% vs 门槛 {(base_up+0.03)*100:.1f}% '
          f' {"✓" if pass_acc else "✗"}   ← 判据已更正，见代码注释')
    print(f'   赢朴素动量基线             : {"✓" if beats_mom else "✗"}'
          f'（{ic_all:+.4f} vs {ic_mom_all:+.4f}）')

    print(f'   ⇒ 择时先决条件{"**成立**，可以继续设计标签/仓位映射" if ok_all else "**不成立**"}')
    if not ok_all:
        print(f'      ⇒ 未来 {a.horizon} 日市场方向用现有 regime 特征**预测不出来**。')
        print(f'         「熊市空仓、牛市梭哈」缺的不是模型结构，是**信号本身不存在**。')
        print(f'         改标签/改损失/加仓位映射都绕不过这一条。')

    out = dict(horizon=a.horizon, embargo=emb, n=int(n), dims=int(p), folds=rows,
               oof=dict(ic=ic_all, acc=acc_all, ic_momentum=ic_mom_all, base_up_ratio=base_up,
                        n=len(T)), bound=dict(hold_all=hold_all, oracle=hold_orc),
               verdict=dict(pass_ic=pass_ic, pass_sign=pass_sign, pass_acc=pass_acc,
                            beats_momentum=beats_mom, precondition_holds=bool(ok_all)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
