import os, sys, argparse, time
import numpy as np
import pandas as pd
import sqlite3
from collections import defaultdict
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.factors.candlestick_pattern_factors import CandlestickPatternFactors

LAGS = [0, 1, 2, 3, 5, 10]
HORIZONS = [7, 15]
IC_START = "2019-01-01"
LOAD_START = "2015-01-01"

def load_ohlc(code, start, end):
    c = sqlite3.connect('database/stock_daily.db')
    df = pd.read_sql_query(
        "SELECT date,open,high,low,close,volume FROM daily_data WHERE code=? AND date>=? AND date<=? ORDER BY date",
        c, params=(code, start, end))
    c.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index().astype(float)
    df = df[~df.index.duplicated(keep='last')]
    return df

def get_base_features(df):
    """返回 date-indexed DataFrame：形态 flags + continuous + OHLC 形态标量。"""
    data = df[['open', 'high', 'low', 'close']]
    allf = CandlestickPatternFactors().calculate_all_candlestick_patterns(data)
    allf.index = df.index
    # OHLC 形态标量（日内结构）
    o, h, l, cl = df['open'], df['high'], df['low'], df['close']
    oc = cl / o - 1.0                      # intraday_ret
    gap = o / df['close'].shift(1) - 1.0   # 跳空
    hl = (h - l)
    close_pos = (cl - l) / hl.replace(0, np.nan)   # 收盘在区间位置
    range_rel = hl / cl                    # 真实波幅占比
    ohlc = pd.DataFrame({
        'm_intraday_ret': oc,
        'm_gap': gap,
        'm_close_pos': close_pos,
        'm_range_rel': range_rel,
    }, index=df.index)
    base = pd.concat([allf.astype(float), ohlc], axis=1)
    return base

def fwd_ret(df, h):
    return df['close'].shift(-h) / df['close'] - 1.0

def rank_ic_panel(panel, yret):
    ics = []
    for d in panel.index:
        a = panel.loc[d].values
        b = yret.loc[d].values
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 50:
            continue
        ic, _ = spearmanr(a[m], b[m])
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) < 10:
        return (np.nan, np.nan, np.nan)
    mean = ics.mean()
    sd = ics.std(ddof=1)
    n = len(ics)
    t = mean / (sd / np.sqrt(n)) if sd > 0 else np.nan
    icir = mean / sd if sd > 0 else np.nan
    return (mean, t, icir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=800)
    ap.add_argument('--end', default="2022-01-01")
    ap.add_argument('--topn', type=int, default=50)
    a = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading codes ...")
    c = sqlite3.connect('database/stock_daily.db')
    codes = [r[0] for r in c.execute(
        "SELECT code, COUNT(*) n FROM daily_data WHERE date>=? AND date<=? GROUP BY code HAVING n>600 ORDER BY n DESC LIMIT ?",
        (LOAD_START, a.end, a.pool))]
    c.close()
    print(f"  {len(codes)} codes")

    # 1) 前向收益面板（date x code）
    print(f"[{time.strftime('%H:%M:%S')}] building forward-return panels ...")
    ypanels = {h: {} for h in HORIZONS}
    for code in codes:
        df = load_ohlc(code, IC_START, a.end)
        for h in HORIZONS:
            ypanels[h][code] = fwd_ret(df, h)
    Y = {h: pd.DataFrame(ypanels[h]) for h in HORIZONS}
    for h in HORIZONS:
        Y[h] = Y[h].reindex(Y[h].index)  # 保持 date 索引
    ic_dates = Y[7].dropna(how='all').index
    for h in HORIZONS:
        Y[h] = Y[h].loc[ic_dates]
    print(f"  ic_dates = {len(ic_dates)}")

    # 2) 滞后分解特征： (base_col, lag) -> {code: Series}
    print(f"[{time.strftime('%H:%M:%S')}] computing lag-resolved candle features ...")
    feat_series = defaultdict(dict)   # (col, lag) -> {code: Series}
    cnt_series = defaultdict(dict)    # (col, 'cnt5'/'cnt10'/'cnt20') -> {code: Series}  (置换不变基线)
    all_base_cols = None
    for ci, code in enumerate(codes):
        df = load_ohlc(code, LOAD_START, a.end)
        base = get_base_features(df)
        if all_base_cols is None:
            all_base_cols = list(base.columns)
        for col in base.columns:
            s = base[col]
            for k in LAGS:
                feat_series[(col, k)][code] = s.shift(k)
        # 形态计数（仅对二值 flags 列，连续量列跳过）
        uniq = set(np.unique(base.dropna().values)) if base.dropna().size else set()
        binary_cols = [col for col in base.columns
                       if col not in ('m_intraday_ret', 'm_gap', 'm_close_pos', 'm_range_rel')
                       and set(np.nan_to_num(base[col].dropna().unique())).issubset({0.0, 1.0})]
        for col in binary_cols:
            for w in (5, 10, 20):
                cnt_series[(col, f'cnt{w}')][code] = (base[col] > 0).astype(float).rolling(w, min_periods=3).mean()
        if (ci + 1) % 200 == 0:
            print(f"  {ci+1}/{len(codes)} codes done")

    # 3) IC 计算
    print(f"[{time.strftime('%H:%M:%S')}] computing IC ...")
    results = []  # (col, tag, h, mean_ic, t, icir)
    # 滞后特征
    keys = list(feat_series.keys())
    for (col, k) in keys:
        panel = pd.DataFrame(feat_series[(col, k)]).loc[ic_dates]
        for h in HORIZONS:
            mean, t, icir = rank_ic_panel(panel, Y[h])
            results.append((col, f'lag{k}', h, mean, t, icir))
    # 计数基线
    for (col, tag) in cnt_series.keys():
        panel = pd.DataFrame(cnt_series[(col, tag)]).loc[ic_dates]
        for h in HORIZONS:
            mean, t, icir = rank_ic_panel(panel, Y[h])
            results.append((col, tag, h, mean, t, icir))

    res_df = pd.DataFrame(results, columns=['col', 'tag', 'h', 'ic', 't', 'icir'])

    # 4) 输出：每个基础特征一行，展示 lag0..lag10 IC7，并标注显著滞后与"顺序信号"
    print(f"\n{'='*120}")
    print(f"K线形态 滞后分解 IC（IC窗口 {IC_START}~{a.end}，{len(codes)} 股，标签 7d/15d 前向收益）")
    print(f"{'='*120}")
    # 只看 7d 标签
    d7 = res_df[res_df['h'] == 7].copy()
    base_cols = [c for c in all_base_cols]
    print(f"{'feature':<26} {'lag0':>8} {'lag1':>8} {'lag2':>8} {'lag3':>8} {'lag5':>8} {'lag10':>8} | {'cnt5':>7} {'cnt10':>7} {'cnt20':>7} | temporal?")
    print('-' * 120)
    temporal_hits = []
    for col in base_cols:
        row = {}
        for _, r in d7[d7['col'] == col].iterrows():
            row[r['tag']] = r
        def g(tag):
            return row[tag]['ic'] if tag in row else np.nan
        def gt(tag):
            return row[tag]['t'] if tag in row else np.nan
        # temporal signature：存在 lag>=1 且 |t|>2.8 且 |ic| 与 lag0 不同号或明显非零
        lags_sig = [k for k in (1,2,3,5,10) if f'lag{k}' in row and abs(row[f'lag{k}']['t']) > 2.8]
        # 且 lag0 不显著 或 有非 lag0 显著
        t0_sig = f'lag0' in row and abs(row['lag0']['t']) > 2.8
        is_temporal = (len(lags_sig) > 0) and (not t0_sig or any(abs(row[f'lag{k}']['ic']) > 0.012 for k in lags_sig))
        flag = 'YES' if is_temporal else ''
        if is_temporal:
            temporal_hits.append((col, lags_sig))
        print(f"{col:<26} {g('lag0'):>8.4f} {g('lag1'):>8.4f} {g('lag2'):>8.4f} {g('lag3'):>8.4f} {g('lag5'):>8.4f} {g('lag10'):>8.4f} | "
              f"{g('cnt5'):>7.4f} {g('cnt10'):>7.4f} {g('cnt20'):>7.4f} | {flag}")

    print(f"\n{'='*120}")
    print(f"TEMPORAL SIGNATURE 命中（滞后>=1 显著且非仅 lag0 的延续）：{len(temporal_hits)} 个")
    for col, lags in temporal_hits:
        print(f"  {col:<26} 显著滞后: {lags}")
    # 汇总
    n_sig_lag0 = d7[d7['tag']=='lag0']['t'].abs() > 2.8
    n_sig_lag1p = d7[d7['tag'].isin([f'lag{k}' for k in (1,2,3,5,10)]) & (d7['t'].abs()>2.8)]
    print(f"\nlag0 显著特征数: {n_sig_lag0.sum()}  |  滞后>=1 显著(特征×滞后组合)数: {len(n_sig_lag1p)}")
    print(f"[{time.strftime('%H:%M:%S')}] done")

if __name__ == '__main__':
    main()
