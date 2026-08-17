"""
零假设检验：同换手节奏下的随机选股基准

动机
----
回测给出 +20% 并不等于模型会选股。一个「每 6.6 天换一批、同时持 20 只」的
高换手框架，在牛市里本身就有相当可观的收益；如果把选股环节换成掷骰子，
收益可能与模型不相上下。此时 +20% 全部来自换手框架与市场 beta，
与「模型学到了什么」无关。

本脚本的做法是**保持策略的换手节奏完全不变**，只把标的替换掉：
读取真实交易记录，取出每一笔的 (buy_date, sell_date, capital_weight)，
再从当日可交易股票池里随机抽一只填进去，按同样的买卖成本结算。
重复 N 次得到零假设收益分布。

判据
----
  策略实际收益  vs  随机分布的分位数
  - 落在 50% 分位附近 -> 选股环节零贡献，收益全来自框架 + beta
  - 落在 95% 分位以上 -> 选股确有超额，且可用单侧 p 值量化
这比「跑赢指数」严格得多，因为它已经扣掉了换手节奏、持仓数量、
时间分布、成本结构这些与选股无关的因素。

用法
----
  python scripts/exp/diag_random_null.py \
      --trades backtest_result/T045/nam/xxx/backtest_trades.csv \
      --n-sims 500 --buy-cost 0.0008 --sell-cost 0.0013
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _db_path() -> str:
    from config.baostock_config import DATABASE_PATH

    return DATABASE_PATH


def load_price_panel(start: str, end: str,
                     exclude_st: bool = True,
                     min_price: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载收益面板与**仅用于入场资格**的布尔面板。

    收益面板保留所有正常交易日的数据：股票买入后即使变 ST 或跌破最低价，
    其后真实涨跌仍必须计入持有期收益，不能被填成 0。
    入场资格才应用当日 ST / 最低价规则，与策略端 PIT 基础风控保持一致。
    """
    sql = (
        "SELECT date, code, pctChg, close, COALESCE(is_st, 0) AS is_st "
        "FROM daily_data WHERE date >= ? AND date <= ? "
        "AND tradestatus = 1 AND pctChg IS NOT NULL ORDER BY date"
    )
    conn = sqlite3.connect(_db_path())
    try:
        df = pd.read_sql_query(sql, conn, params=[start, end])
    finally:
        conn.close()
    df['date'] = df['date'].astype(str)
    ret_wide = df.pivot_table(index='date', columns='code', values='pctChg', aggfunc='last') / 100.0
    eligible = np.isfinite(pd.to_numeric(df['pctChg'], errors='coerce'))
    if exclude_st:
        eligible &= pd.to_numeric(df['is_st'], errors='coerce').fillna(0).eq(0)
    if min_price is not None:
        eligible &= pd.to_numeric(df['close'], errors='coerce').ge(float(min_price))
    elig_df = df[['date', 'code']].copy()
    elig_df['eligible'] = eligible.to_numpy(dtype=bool)
    eligibility = elig_df.pivot_table(
        index='date', columns='code', values='eligible', aggfunc='last', fill_value=False
    ).reindex(index=ret_wide.index, columns=ret_wide.columns, fill_value=False).astype(bool)
    return ret_wide, eligibility


def _norm_code(code: str) -> str:
    """交易记录里是裸 6 位码，数据库里是 sh.600000 形式。"""
    c = str(code).strip()
    if '.' in c:
        return c
    c = c.zfill(6)
    return ('sh.' if c[0] in '65' else 'sz.') + c


def simulate(trades: pd.DataFrame,
             ret_wide: pd.DataFrame,
             eligibility: pd.DataFrame,
             buy_cost: float,
             sell_cost: float,
             n_sims: int,
             seed: int = 42) -> Dict[str, object]:
    """按真实换手节奏做随机选股蒙特卡洛。

    收益合成方式：每笔交易占用资金权重 w（= 1/max_positions 的近似，
    直接用交易记录里的实际分配），持有期收益 r 由随机标的在同期的
    复利收益给出，净收益 = (1+r)*(1-sell_cost) / (1+buy_cost) - 1。
    组合总收益按「每笔独立、权重加权求和后逐笔复利」近似——
    与真实回测的差别只在现金再投资的二阶项，不影响分位判断。
    """
    rng = np.random.default_rng(seed)
    dates = list(ret_wide.index)
    date_idx = {d: i for i, d in enumerate(dates)}
    codes = np.array(ret_wide.columns)
    mat = ret_wide.to_numpy(dtype=np.float32)
    entry_eligible = eligibility.reindex(
        index=ret_wide.index, columns=ret_wide.columns, fill_value=False
    ).to_numpy(dtype=bool)

    # 预处理每笔交易 -> (start_i, end_i, weight)。
    # 入口池必须排除当日真实持仓标的，避免随机基准在同一时点重复抽到
    # 策略已持有股票；这是组合层零假设的必要约束，而不是简单独立抽样。
    legs: List[Tuple[int, int, float, str]] = []
    for _, r in trades.iterrows():
        bd, sd = str(r['buy_date']), str(r['sell_date'])
        if bd not in date_idx or sd not in date_idx:
            continue
        i0, i1 = date_idx[bd], date_idx[sd]
        if i1 <= i0:
            i1 = i0 + 1
        # 权重：用买入市值 / 该笔买入时的名义总资金 1.0 的近似
        w = float(r.get('capital_weight', np.nan))
        if not np.isfinite(w):
            w = float(r['shares']) * float(r['buy_price'])
        legs.append((i0, i1, w, _norm_code(r.get('stock_code', ''))))
    if not legs:
        raise SystemExit('没有可用交易腿')

    # 每个交易日的可选股票索引
    pool_cache: Dict[int, np.ndarray] = {}

    def pool(i: int) -> np.ndarray:
        if i not in pool_cache:
            pool_cache[i] = np.flatnonzero(entry_eligible[i])
        return pool_cache[i]

    logret = np.log1p(np.nan_to_num(mat, nan=0.0))
    cs = np.vstack([np.zeros((1, logret.shape[1]), dtype=np.float64),
                    np.cumsum(logret, axis=0)])  # cs[i] = sum of logret[:i]

    sims = np.zeros(n_sims, dtype=float)
    for s in range(n_sims):
        total = 0.0
        # 每次模拟维护区间内的持仓，按真实交易腿的开仓/平仓时点更新。
        # 交易记录通常按开仓日排序；显式排序保证工具不依赖 CSV 行序。
        active = []
        for (i0, i1, w, _actual_code) in sorted(legs, key=lambda x: (x[0], x[1])):
            active = [pos for pos in active if pos[1] > i0]
            p = pool(i0)
            if len(p) == 0:
                continue
            occupied = {pos[2] for pos in active}
            # 候选池约 5000，只排除同时持有的 <=20 个索引；避免每条腿构造
            # Python list 并逐元素做 membership（800 次模拟下是主要性能瓶颈）。
            if occupied:
                available = p[~np.isin(p, np.fromiter(occupied, dtype=np.int64), assume_unique=False)]
            else:
                available = p
            if len(available) == 0:
                # 若真实组合已满且随机池无法无重复填充，保守跳过该腿，
                # 不用重复持仓虚构分散化收益。
                continue
            j = available[rng.integers(len(available))]
            active.append((i0, i1, int(j)))
            # 持有期收益：买入日次日起至卖出日收盘
            gross = float(np.exp(cs[i1 + 1, j] - cs[i0 + 1, j]) - 1.0)
            net = (1.0 + gross) * (1.0 - sell_cost) / (1.0 + buy_cost) - 1.0
            total += w * net
        sims[s] = total
    return {
        'sims': sims,
        'n_legs': len(legs),
        'total_weight': float(sum(w for _, _, w, _ in legs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', required=True)
    ap.add_argument('--n-sims', type=int, default=300)
    ap.add_argument('--buy-cost', type=float, default=0.0008)
    ap.add_argument('--sell-cost', type=float, default=0.0013)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='diagnose_output/random_null.json')
    args = ap.parse_args()

    tr = pd.read_csv(args.trades, encoding='utf-8-sig')
    tr['buy_date'] = tr['buy_date'].astype(str)
    tr['sell_date'] = tr['sell_date'].astype(str)
    tr['capital_weight'] = tr['shares'].astype(float) * tr['buy_price'].astype(float)

    start, end = tr['buy_date'].min(), tr['sell_date'].max()
    print(f'交易记录: {len(tr)} 笔, {start} ~ {end}')
    print(f'平均持仓天数: {tr["holding_days"].mean():.2f}   '
          f'平均单笔权重: {tr["capital_weight"].mean():.4f}')

    ret_wide, eligibility = load_price_panel(start, end)
    print(f'价格面板: {ret_wide.shape[0]} 日 × {ret_wide.shape[1]} 股')

    # 真实策略在同一成本口径下的收益（用记录里的毛收益重算，剥离旧成本）
    # pnl_pct 是含旧成本的净收益；这里用 buy/sell 价重算毛收益再套新成本
    gross = tr['sell_price'].astype(float) / tr['buy_price'].astype(float) - 1.0
    net_new = (1.0 + gross) * (1.0 - args.sell_cost) / (1.0 + args.buy_cost) - 1.0
    actual = float((tr['capital_weight'] * net_new).sum())
    print(f'策略（新成本口径，权重加权求和）: {actual * 100:+.2f}%')

    res = simulate(
        tr, ret_wide, eligibility,
        args.buy_cost, args.sell_cost, args.n_sims, args.seed,
    )
    sims = res['sims']
    pct = float((sims < actual).mean() * 100)
    print('\n' + '=' * 72)
    print(f'随机选股零假设分布（{args.n_sims} 次，{res["n_legs"]} 条交易腿）')
    print('=' * 72)
    for q in (5, 25, 50, 75, 95):
        print(f'  P{q:<3d}: {np.percentile(sims, q) * 100:+8.2f}%')
    print(f'  均值 : {sims.mean() * 100:+8.2f}%   标准差 {sims.std(ddof=1) * 100:.2f}%')
    print('-' * 72)
    z = (actual - sims.mean()) / sims.std(ddof=1) if sims.std(ddof=1) > 0 else float('nan')
    print(f'  策略 : {actual * 100:+8.2f}%  ->  分位 {pct:.1f}%   z = {z:+.2f}')
    print(f'  单侧 p = {(sims >= actual).mean():.4f}')
    print('=' * 72)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({
            'trades_file': args.trades,
            'buy_cost': args.buy_cost, 'sell_cost': args.sell_cost,
            'n_sims': args.n_sims, 'n_legs': res['n_legs'],
            'actual_return_pct': actual * 100,
            'null_mean_pct': float(sims.mean() * 100),
            'null_std_pct': float(sims.std(ddof=1) * 100),
            'null_percentiles': {f'p{q}': float(np.percentile(sims, q) * 100)
                                 for q in (5, 25, 50, 75, 95)},
            'percentile_of_actual': pct,
            'z_score': float(z),
            'p_value_one_sided': float((sims >= actual).mean()),
        }, f, ensure_ascii=False, indent=2)
    print(f'已保存: {args.out}')


if __name__ == '__main__':
    main()

