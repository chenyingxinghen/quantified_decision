"""T079 / E10 第一步：把**业绩预告**（baostock ``query_forecast_report``）灌进库。

为什么是这条：E7/E8 已经把「从现有 OHLCV/基本面再派生新列」这条路走死——
去掉已有列明确变差、加派生列落在噪声带内（台账 E8 判定）。瓶颈是**信息**。
而库里目前只有行情 + 四张财务能力表 + 行业 + 市场情绪，`performance_forecast`
表在 `baostock_fetcher.py` 里有建表分支、`fetch_performance_forecast` 也早就写好了，
但 `FINANCE_TABLES` 从未包含它 —— **抓取代码齐备、数据一次都没落库**。

业绩预告是真正的新信息：它是**报表公布之前**的盈利消息（现有 `sue` 用的是已公布的
净利同比，是事后量）。A 股预告的披露日效应是公开文献里最稳的截面异象之一。

PIT 纪律：只用 ``pub_date``（预告实际披露日）对齐，绝不用 ``stat_date``（报告期）。
本脚本只负责落库，不做任何特征；特征在 T079 第二步按 pub_date 前向填充。

用法（可断点续跑，已抓过的 code 直接跳过）:
  python scripts/ingest_performance_forecast.py --start 2010-01-01
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DDL = '''
CREATE TABLE IF NOT EXISTS performance_forecast (
    code TEXT, pub_date TEXT, stat_date TEXT,
    profitForcastExpPubDate TEXT, profitForcastType TEXT,
    profitForcastAbstract TEXT,
    profitForcastChgPctUp REAL, profitForcastChgPctDwn REAL,
    PRIMARY KEY (code, pub_date, stat_date)
)
'''
COLS = ['code', 'pub_date', 'stat_date', 'profitForcastExpPubDate',
        'profitForcastType', 'profitForcastAbstract',
        'profitForcastChgPctUp', 'profitForcastChgPctDwn']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2010-01-01')
    ap.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'))
    ap.add_argument('--limit', type=int, default=0, help='只抓前 N 只，用于冒烟')
    a = ap.parse_args()

    import pandas as pd
    from config.baostock_config import FINANCE_DB_PATH
    from core.data.baostock_fetcher_methods import fetch_performance_forecast
    from core.data.baostock_main import BaostockDataManager

    conn = sqlite3.connect(FINANCE_DB_PATH, timeout=60.0)
    conn.execute(DDL)
    conn.execute('CREATE INDEX IF NOT EXISTS idx_pf_code_pub '
                 'ON performance_forecast (code, pub_date)')
    conn.commit()
    # 断点续跑：已有记录的 code 跳过。注意「无预告」的股票会每次重抓，
    # 用一张进度表记下已抓过的 code，避免重复消耗配额。
    conn.execute('CREATE TABLE IF NOT EXISTS performance_forecast_done '
                 '(code TEXT PRIMARY KEY, n INTEGER, ts TEXT)')
    conn.commit()
    done = {r[0] for r in conn.execute('SELECT code FROM performance_forecast_done')}

    mgr = BaostockDataManager()
    codes = mgr.get_stock_list_from_db()['code'].tolist()
    mgr.close()
    if a.limit:
        codes = codes[:a.limit]
    todo = [c for c in codes if c not in done]
    print(f'共 {len(codes)} 只，已完成 {len(codes) - len(todo)}，待抓 {len(todo)}')

    t0, ok, rows_total, fail = time.time(), 0, 0, 0
    # 探针：fetch 在失败时也返回空 DataFrame，和「这只票确实没有预告」无法区分。
    # 所以先用一只必然有预告的票验活；探针空 = 登录/配额/接口坏了，直接中止，
    # 否则会把 5500 只全标成「已抓、0 行」（第一次跑就踩到了：baostock 返回
    # 「黑名单用户」，脚本却当成没有预告）。
    probe = fetch_performance_forecast('000001', a.start, a.end)
    if probe is None or probe.empty:
        print('[ABORT] 探针 000001 无返回 —— baostock 登录/配额/接口异常，'
              '本次不写入任何进度。')
        conn.close()
        return
    # 第二道探针：接口有返回、但下游要的 pub_date 列不存在（baostock 的
    # forecast 接口用 profitForcastExpPubDate 而非 pubDate），会被
    # dropna(subset=['pub_date']) 静默清成 0 行，把 5500 只全标成「已抓、0 行」。
    # 这个坑真踩过一次，所以在这里硬拦。
    missing = [c for c in ('pub_date', 'stat_date') if c not in probe.columns]
    if missing:
        print(f'[ABORT] 探针返回 {len(probe)} 行但缺列 {missing} —— '
              f'实际列 {list(probe.columns)}；接口字段名变了，先修 '
              f'fetch_performance_forecast 的改名逻辑，不要往进度表里写东西。')
        conn.close()
        return
    print(f'  探针 000001: {len(probe)} 条预告，接口正常')

    empty_streak = 0
    for i, code in enumerate(todo, 1):
        try:
            df = fetch_performance_forecast(code, a.start, a.end)
        except Exception as e:                       # 配额/网络异常：停下来而不是静默跳过
            print(f'  [ABORT] {code}: {e}')
            break
        n = 0
        if df is not None and not df.empty:
            for c in COLS:
                if c not in df.columns:
                    df[c] = None
            df['code'] = code
            for c in ('profitForcastChgPctUp', 'profitForcastChgPctDwn'):
                df[c] = pd.to_numeric(df[c], errors='coerce')
            sub = df[COLS].dropna(subset=['pub_date'])
            conn.executemany(
                f'INSERT OR REPLACE INTO performance_forecast ({",".join(COLS)}) '
                f'VALUES ({",".join("?" * len(COLS))})',
                sub.itertuples(index=False, name=None))
            n = len(sub)
        conn.execute('INSERT OR REPLACE INTO performance_forecast_done VALUES (?,?,?)',
                     (code, n, datetime.now().isoformat(timespec='seconds')))
        conn.commit()
        rows_total += n
        ok += 1
        empty_streak = empty_streak + 1 if n == 0 else 0
        if empty_streak >= 40:
            print(f'  [ABORT] 连续 {empty_streak} 只返回空 —— 大概率是接口/配额坏了，'
                  f'而不是这些票都没预告。停止以免污染进度表。')
            break
        if i % 200 == 0:
            el = time.time() - t0
            print(f'  {i}/{len(todo)}  累计 {rows_total} 行  '
                  f'{el / i:.2f}s/只  预计剩余 {(len(todo) - i) * el / i / 60:.1f} min')

    total = conn.execute('SELECT count(*) FROM performance_forecast').fetchone()[0]
    ncode = conn.execute('SELECT count(DISTINCT code) FROM performance_forecast').fetchone()[0]
    print(f'\n本次成功 {ok} 只，新增/更新 {rows_total} 行，失败 {fail}')
    print(f'库内合计 {total} 行，覆盖 {ncode} 只股票')
    conn.close()


if __name__ == '__main__':
    main()
