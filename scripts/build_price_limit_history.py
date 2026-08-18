"""从行情数据**实测**每日各板块的涨跌停限额，落库成 (date, board, is_st) → limit。

为什么不查表
------------
`config.MARKET_LIMITS` 是静态查表（main 0.098 / gem_star 0.198 / bj 0.295 / st 0.05），
没有时间维度。但限额在历史上变过多次（创业板 2020-08-24 由 10% 改 20%、北交所设立、
ST 规则调整……），维护一张规则表就是在维护一个必然过期的东西 —— 用户 2026-08 指出
ST 板已回到 10%，而代码仍写 5%。

实测判据
--------
涨停价是 ``round(preclose × (1+L), 2)`` 这个**精确值**，且当日 ``close == high``。
随机股票撞不上该精确价位，真涨停会撞出一大堆。对每个 (交易日, 板块, is_st)：
对候选 L ∈ {5%, 10%, 20%, 30%} 数精确命中，取命中最多者；命中为 0 的日子留空，
最后按组前向填充（限额是分段常数，中间的清淡日没有信息，不该猜）。

自检：估计序列应在已知变更日出现台阶。实测创业板恰好在 2020-08-24 当天
由 0.1 切到 0.2（前一日 0.1 命中 22 次、0.2 零次；当日 0.1 归零、0.2 出现），
该日期**不是输入而是输出** —— 这就是方法有效的证据。

用法:
  python scripts/build_price_limit_history.py            # 全量重建
  python scripts/build_price_limit_history.py --report   # 只打印台阶，不写库
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DAILY_DB = os.path.join(ROOT, 'database', 'stock_daily.db')
CANDIDATES = (0.05, 0.10, 0.20, 0.30)
# 命中数下限：低于它认为当日无涨停样本，不做判定（留空后前向填充）
MIN_HITS = 3
# 判定优势倍数：赢家必须比次优候选多这么多倍命中，否则视为无判定。
# 清淡日只有零星涨停时，个别股票的收盘价会恰好撞上 preclose×1.05 的精确价位，
# 制造出「创业板某天限额 5%」这种假台阶（实测 2022-10-12、2026-04-08 等）。
MIN_MARGIN = 2.0
# 持续性：限额是制度，不会只变一两天又变回去。短于该天数的「台阶」判为噪声，
# 用前一段的值覆盖。
MIN_RUN_DAYS = 15

DDL = '''
CREATE TABLE IF NOT EXISTS price_limit_history (
    date TEXT, board TEXT, is_st INTEGER, limit_pct REAL,
    n_hits INTEGER, inferred INTEGER,
    PRIMARY KEY (date, board, is_st)
)
'''


def board_of(code: str) -> str:
    if code.startswith(('300', '301')):
        return 'gem'
    if code.startswith(('688', '689')):
        return 'star'
    if code.startswith(('43', '83', '87', '88')):
        return 'bj'
    return 'main'


def estimate(df: pd.DataFrame) -> pd.DataFrame:
    """df: code/date/preclose/high/close/is_st -> 每 (date,board,is_st) 一行估计。"""
    df = df[(df['preclose'] > 0) & df['close'].notna() & df['high'].notna()].copy()
    # 只用收在最高价的样本（真涨停的必要条件），大幅压低随机撞价
    df = df[(df['close'] - df['high']).abs() < 1e-6]
    if df.empty:
        return pd.DataFrame()
    df['board'] = df['code'].map(board_of)
    df['is_st'] = df['is_st'].fillna(0).astype(int)

    for L in CANDIDATES:
        limit_price = np.round(df['preclose'].to_numpy() * (1 + L), 2)
        df[f'h{L}'] = (np.abs(df['close'].to_numpy() - limit_price) < 1e-6).astype(np.int32)

    g = df.groupby(['date', 'board', 'is_st'], sort=False)[[f'h{L}' for L in CANDIDATES]].sum()
    arr = g.to_numpy()
    order = np.sort(arr, axis=1)
    best_hits = order[:, -1]
    second = order[:, -2]
    best_idx = arr.argmax(axis=1)
    out = g.reset_index()[['date', 'board', 'is_st']].copy()
    out['limit_pct'] = [CANDIDATES[i] for i in best_idx]
    out['n_hits'] = best_hits
    # 命中太少、或赢得不够干净 → 不判定（留空，交给前向填充）
    weak = (best_hits < MIN_HITS) | (best_hits < MIN_MARGIN * np.maximum(second, 1))
    out.loc[weak, 'limit_pct'] = np.nan
    return out


def _despike(s: pd.Series, dates: pd.Series) -> pd.Series:
    """把短于 MIN_RUN_DAYS 的「台阶」判为噪声，用前一段的值覆盖。

    限额是制度参数，不存在只持续几天又改回去的情况；这类短段一律来自
    清淡日的偶然精确撞价。首段不做覆盖（它没有「前一段」可依）。
    """
    v = s.to_numpy(copy=True)
    if len(v) == 0:
        return s
    # 段边界
    starts = [0] + list(np.flatnonzero(v[1:] != v[:-1]) + 1)
    ends = starts[1:] + [len(v)]
    for i, (a, b) in enumerate(zip(starts, ends)):
        if i == 0:
            continue
        if (b - a) < MIN_RUN_DAYS:
            v[a:b] = v[a - 1]
    return pd.Series(v, index=s.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true', help='只打印台阶，不写库')
    a = ap.parse_args()

    conn = sqlite3.connect(DAILY_DB, timeout=120.0)
    years = [r[0] for r in conn.execute(
        'SELECT DISTINCT substr(date,1,4) FROM daily_data ORDER BY 1')]
    print(f'年份 {years[0]}~{years[-1]}（{len(years)} 年），逐年估计…')

    parts = []
    for y in years:
        df = pd.read_sql_query(
            'SELECT code,date,preclose,high,close,is_st FROM daily_data '
            "WHERE substr(date,1,4)=? AND volume>0", conn, params=(y,))
        if df.empty:
            continue
        est = estimate(df)
        if not est.empty:
            parts.append(est)
        print(f'  {y}: {len(df):>8} 行 -> {len(est):>5} 组估计')
    conn.close()

    full = pd.concat(parts, ignore_index=True).sort_values(['board', 'is_st', 'date'])
    # 限额是分段常数：无涨停样本的日子不猜，按组前向填充
    full['inferred'] = full['limit_pct'].isna().astype(int)
    full['limit_pct'] = full.groupby(['board', 'is_st'])['limit_pct'].ffill()
    full = full.dropna(subset=['limit_pct'])
    # 去掉短命台阶（清淡日偶然撞价造成）
    full['limit_pct'] = (full.groupby(['board', 'is_st'], group_keys=False)
                             .apply(lambda g: _despike(g['limit_pct'], g['date'])))

    print('\n=== 估计出的台阶（每次变更打印一行）===')
    for (bd, st), g in full.groupby(['board', 'is_st']):
        g = g.sort_values('date')
        chg = g[g['limit_pct'] != g['limit_pct'].shift()]
        spans = []
        for i, row in enumerate(chg.itertuples()):
            end = chg.iloc[i + 1].date if i + 1 < len(chg) else g.iloc[-1].date
            spans.append(f'{row.date}~{end}: {row.limit_pct:.0%}')
        print(f'  {bd:5s} is_st={st}: ' + ' | '.join(spans))

    if a.report:
        print('\n--report：未写库')
        return 0

    conn = sqlite3.connect(DAILY_DB, timeout=120.0)
    conn.execute(DDL)
    conn.execute('DELETE FROM price_limit_history')
    conn.executemany(
        'INSERT OR REPLACE INTO price_limit_history VALUES (?,?,?,?,?,?)',
        full[['date', 'board', 'is_st', 'limit_pct', 'n_hits', 'inferred']]
            .astype({'n_hits': int, 'inferred': int}).itertuples(index=False, name=None))
    conn.commit()
    n = conn.execute('SELECT COUNT(*) FROM price_limit_history').fetchone()[0]
    conn.close()
    print(f'\n已写入 price_limit_history: {n} 行')
    return 0


if __name__ == '__main__':
    sys.exit(main())
