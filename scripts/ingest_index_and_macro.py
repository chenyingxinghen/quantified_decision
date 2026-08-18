"""T115/T116 数据落库：指数日线 + 货币供应月度（+ 成分股 API 历史深度探针）。

背景（2026-08-18 数据缺口讨论定案）：
- 指数行情：库里此前完全没有。用途是**逐股截面列**（beta/相对强弱/特异波动），
  不是给门控喂 regime——门控轴已终审关闭（T108/T111）。
- 货币供应：M1-M2 剪刀差是标量门终检（T116）唯一的外生条件化信号。
  月度数据发布滞后约次月 10~15 日，PIT 对齐一律按 ``statMonth 次月 15 日``
  才可见（保守），特征侧不得直接用 statMonth 当可见日。
- 成分股：baostock 的 query_hs300_stocks(date) 历史深度未经验证。本脚本只探针
  （打印若干历史日期的返回行数），**不建表不落库**——若只有近期快照，
  拿它做历史特征就是幸存者偏差。

沿用 E10 预告落库的三条纪律：
1. 探针先行：接口空返回 = 登录/配额/接口坏，立即中止，不写任何进度；
2. 字段名硬校验：缺关键列直接 ABORT 并打印实际列名（query_forecast_report
   的 pubDate 改名坑真踩过一次）；
3. 幂等可重跑：INSERT OR REPLACE + 主键，重跑不产生重复行。

用法:
  python scripts/ingest_index_and_macro.py --index --money --probe-constituents
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 六个核心指数：上证综指 / 深证成指 / 沪深300 / 中证500 / 上证50 / 创业板指
INDEX_CODES = ['sh.000001', 'sz.399001', 'sh.000300', 'sh.000905',
               'sh.000016', 'sz.399006']

INDEX_DDL = '''
CREATE TABLE IF NOT EXISTS index_daily (
    code TEXT, date TEXT,
    open REAL, high REAL, low REAL, close REAL, preclose REAL,
    volume REAL, amount REAL, pctChg REAL,
    PRIMARY KEY (code, date)
)
'''

MONEY_DDL = '''
CREATE TABLE IF NOT EXISTS macro_money_supply (
    statMonth TEXT PRIMARY KEY,
    m0Month REAL, m0YOY REAL, m0ChainRelative REAL,
    m1Month REAL, m1YOY REAL, m1ChainRelative REAL,
    m2Month REAL, m2YOY REAL, m2ChainRelative REAL,
    fetched_at TEXT
)
'''


def _f(x):
    try:
        return float(x) if x not in ('', None) else None
    except (TypeError, ValueError):
        return None


def ingest_index(start: str, end: str) -> bool:
    from config.baostock_config import DATABASE_PATH
    from core.data.baostock_fetcher_methods import _bs_query

    conn = sqlite3.connect(DATABASE_PATH, timeout=60.0)
    conn.execute(INDEX_DDL)
    conn.execute('CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily (date)')
    conn.commit()

    fields = 'date,code,open,high,low,close,preclose,volume,amount,pctChg'
    ok = True
    for code in INDEX_CODES:
        rs = _bs_query('query_history_k_data_plus', code=code, fields=fields,
                       start_date=start, end_date=end, frequency='d')
        if rs is None or not rs.data:
            print(f'[ABORT] 指数 {code} 无返回 —— 接口/登录异常，中止指数落库')
            ok = False
            break
        cols = list(rs.fields)
        need = ['date', 'code', 'close', 'preclose', 'pctChg']
        missing = [c for c in need if c not in cols]
        if missing:
            print(f'[ABORT] 指数 {code} 返回缺列 {missing}，实际列 {cols}')
            ok = False
            break
        idx = {c: cols.index(c) for c in cols}
        rows = []
        for r in rs.data:
            rows.append((r[idx['code']], r[idx['date']],
                         _f(r[idx.get('open', 0)]), _f(r[idx.get('high', 0)]),
                         _f(r[idx.get('low', 0)]), _f(r[idx['close']]),
                         _f(r[idx['preclose']]), _f(r[idx.get('volume', 0)]),
                         _f(r[idx.get('amount', 0)]), _f(r[idx['pctChg']])))
        conn.executemany('INSERT OR REPLACE INTO index_daily VALUES (?,?,?,?,?,?,?,?,?,?)', rows)
        conn.commit()
        span = (rows[0][1], rows[-1][1]) if rows else ('-', '-')
        print(f'  {code}: {len(rows)} 行  {span[0]} → {span[1]}')
    n = conn.execute('SELECT COUNT(*), COUNT(DISTINCT code) FROM index_daily').fetchone()
    print(f'index_daily 合计: {n[0]} 行 / {n[1]} 个指数')
    conn.close()
    return ok


def ingest_money(start_ym: str, end_ym: str) -> bool:
    from config.baostock_config import META_DB_PATH
    from core.data.baostock_fetcher_methods import _bs_query

    rs = _bs_query('query_money_supply_data_month', start_date=start_ym, end_date=end_ym)
    if rs is None or not rs.data:
        print('[ABORT] 货币供应无返回 —— 接口/登录异常，不写库')
        return False
    cols = list(rs.fields)
    # E10 字段名硬校验：剪刀差必需 m1YOY/m2YOY。
    # 首跑实测：接口把年月拆成 statYear + statMonth 两列（statMonth 只有 '01'..'12'），
    # 只存 statMonth 会让主键把全部历史坍缩成 12 行 —— 必须组合成 'YYYY-MM'。
    need = ['statYear', 'statMonth', 'm1YOY', 'm2YOY']
    missing = [c for c in need if c not in cols]
    if missing:
        print(f'[ABORT] 货币供应返回 {len(rs.data)} 行但缺列 {missing} —— 实际列 {cols}；'
              f'接口字段名与预期不符，先修本脚本的列映射，不要落库。')
        return False
    idx = {c: cols.index(c) for c in cols}
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    keep = ['m0Month', 'm0YOY', 'm0ChainRelative',
            'm1Month', 'm1YOY', 'm1ChainRelative',
            'm2Month', 'm2YOY', 'm2ChainRelative']
    rows = []
    for r in rs.data:
        ym = f"{r[idx['statYear']]}-{str(r[idx['statMonth']]).zfill(2)}"
        vals = [r[idx[c]] if c in idx else None for c in keep]
        rows.append((ym,) + tuple(_f(v) for v in vals) + (now,))

    conn = sqlite3.connect(META_DB_PATH, timeout=60.0)
    conn.execute('DROP TABLE IF EXISTS macro_money_supply')
    conn.execute(MONEY_DDL)
    conn.executemany('INSERT OR REPLACE INTO macro_money_supply VALUES (?,?,?,?,?,?,?,?,?,?,?)', rows)
    conn.commit()
    chk = conn.execute('SELECT MIN(statMonth), MAX(statMonth), COUNT(*) FROM macro_money_supply').fetchone()
    nn = conn.execute('SELECT COUNT(*) FROM macro_money_supply WHERE m1YOY IS NOT NULL AND m2YOY IS NOT NULL').fetchone()[0]
    print(f'macro_money_supply: {chk[2]} 行  {chk[0]} → {chk[1]}  (m1YOY&m2YOY 非空 {nn})')
    conn.close()
    return True


def probe_constituents():
    """只探针不落库：验证 query_hs300_stocks(date) 的历史深度。"""
    from core.data.baostock_fetcher_methods import _bs_query
    print('成分股 API 历史深度探针（只打印，不建表）:')
    for d in ['2010-06-30', '2015-06-30', '2020-06-30', '2024-06-28', '2026-08-14']:
        for api in ['query_hs300_stocks', 'query_zz500_stocks', 'query_sz50_stocks']:
            rs = _bs_query(api, date=d)
            n = len(rs.data) if rs is not None else -1
            print(f'  {api}({d}): {n} 行')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', action='store_true')
    ap.add_argument('--money', action='store_true')
    ap.add_argument('--probe-constituents', action='store_true')
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'))
    a = ap.parse_args()
    if not (a.index or a.money or a.probe_constituents):
        ap.error('至少指定 --index / --money / --probe-constituents 之一')

    from core.data.baostock_fetcher import BaostockFetcher
    if not BaostockFetcher._bs_login():
        print('[ABORT] baostock 登录失败')
        return 1
    try:
        rc = 0
        if a.index:
            rc |= 0 if ingest_index(a.start, a.end) else 1
        if a.money:
            rc |= 0 if ingest_money(a.start[:7], a.end[:7]) else 1
        if a.probe_constituents:
            probe_constituents()
        return rc
    finally:
        # BaostockFetcher.logout 是实例方法但只操作类态；这里直接注销并释放会话锁
        try:
            import baostock as bs
            bs.logout()
            BaostockFetcher._global_bs_logged_in = False
            from core.data.baostock_fetcher import _release_bs_session_lock
            _release_bs_session_lock()
            print('Baostock 已注销')
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main())
