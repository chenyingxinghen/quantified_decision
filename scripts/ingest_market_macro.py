"""阶段0：市场级外生数据落库（两融 / 期指升贴水 / 宏观利率）。

背景
----
现有 236 因子全部是「价格派生 + 财报派生」，无一列真正外生：
- 市场宽度（up_ratio/breadth_ma20）是**全市场价格聚合**，本质还是价格；
- idx_* 是指数价格派生；
- fc_* 是公司自身预告（有前瞻性但覆盖率仅 19~39%）。

阶段0 补的是**价格之外的市场参与者和货币政策信号**：
  1. 融资融券余额 —— 杠杆需求 / 风险偏好（全市场日频）
  2. 股指期货升贴水 —— 机构对冲需求 / 市场预期（负基差=悲观）
  3. 宏观利率（SHIBOR / LPR / 国债收益率）—— 货币环境

**北向资金已剔除**（2026-08-24 决策）：
沪深港通自 2024-08-19 起停披露日度净买入（改为仅季度披露持股）。探针确认
akshare 接口 2024-08-19 后该列恒 NaN。一个训练窗口有值、实盘窗口恒 NULL
的特征会造成训练/推理分布错配（T043 教训），且永远无法更新 → 不上线。

PIT 纪律（与 E10 一致，落库阶段只做「对」的存储，特征阶段再做前向填充）：
- 两融余额：T 日盘后披露 → T 日可得。
- 期指升贴水：basis = 主力收盘 / 现货指数收盘 - 1，单位 1（0.001=贴水 0.1%）。
  IF→000300，IH→000016，IC→000905（现货取 index_daily）。T 日可得。
- SHIBOR / 国债收益率：T 日发布 → T 日可得。
- LPR：每月 20 日 9:30 发布 → 当日可得（月度序列，特征阶段前向填充）。

写入纪律（E10 三纪律 + 本次踩坑修正）：
  1. 探针先行：接口空返回 = 接口/字段变动，立即中止，不写任何进度；
  2. 字段名硬校验：缺关键列直接 ABORT 并打印实际列名；
  3. 幂等可重跑：**全部用 ON CONFLICT(date) DO UPDATE 按列更新，绝不用
     INSERT OR REPLACE** —— REPLACE 是删整行再插，多个来源写同一张宽表时
     后跑的段会清掉先跑段的列（2026-08-24 实测踩坑：--only north 把两融/
     期指/SHIBOR 的行整行替换掉了，非空行数骤降）。

用法:
  python scripts/ingest_market_macro.py              # 全量落库（不删已有）
  python scripts/ingest_market_macro.py --reset     # DROP 表后全量重灌
  python scripts/ingest_market_macro.py --only margin
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

DDL = '''
CREATE TABLE IF NOT EXISTS market_macro_daily (
    date TEXT PRIMARY KEY,
    -- 两融（亿元）
    margin_balance REAL,
    margin_fin_balance REAL,
    margin_sec_balance REAL,
    margin_fin_buy REAL,
    -- 期指升贴水（1.0 = 100%；IF→000300 / IH→000016 / IC→000905）
    basis_if REAL,
    basis_ih REAL,
    basis_ic REAL,
    -- 宏观利率（%）
    shibor_on REAL,
    shibor_1w REAL,
    shibor_3m REAL,
    lpr_1y REAL,
    cn10y REAL,
    fetched_at TEXT
)
'''


def probe_abort(msg: str, df=None):
    print(f'[ABORT] {msg}')
    if df is not None and len(df):
        print(f'  实际列: {list(df.columns)}')
    sys.exit(1)


def upsert(conn: sqlite3.Connection, col_map: dict, rows, now: str):
    """按列 upsert。col_map: {列名: 值}。只更新给定列，绝不删行。"""
    cols = ['date'] + list(col_map.keys()) + ['fetched_at']
    placeholders = ','.join('?' * len(cols))
    update_sql = ','.join(f'{c}=excluded.{c}' for c in col_map)
    sql = (f'INSERT INTO market_macro_daily ({",".join(cols)}) VALUES ({placeholders}) '
           f'ON CONFLICT(date) DO UPDATE SET {update_sql}')
    conn.executemany(sql, [(r['date'],) + tuple(r[c] for c in col_map) + (now,)
                           for r in rows])
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['margin', 'basis', 'rates'],
                    default=None, help='只落某一类（默认全部）')
    ap.add_argument('--reset', action='store_true', help='DROP 表后全量重灌')
    a = ap.parse_args()

    import pandas as pd
    import akshare as ak
    from config.baostock_config import META_DB_PATH, DATABASE_PATH

    conn = sqlite3.connect(META_DB_PATH, timeout=60.0)
    if a.reset:
        print('[RESET] 重建 market_macro_daily')
        conn.execute('DROP TABLE IF EXISTS market_macro_daily')
    conn.execute(DDL)
    conn.commit()

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── 1. 融资融券 ──────────────────────────────────────────────────────
    if a.only in (None, 'margin'):
        print('=== 1. 融资融券 ===')
        df = ak.stock_margin_account_info()
        if df is None or len(df) == 0:
            probe_abort('两融探针空返回 —— akshare 接口变动，中止', df)
        need = ['日期', '融资余额', '融券余额', '融资买入额']
        missing = [c for c in need if c not in df.columns]
        if missing:
            probe_abort(f'两融返回缺列 {missing}', df)
        df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
        fin = pd.to_numeric(df['融资余额'], errors='coerce')
        sec = pd.to_numeric(df['融券余额'], errors='coerce')
        fbuy = pd.to_numeric(df['融资买入额'], errors='coerce')
        rows = []
        for i, d in enumerate(df['日期']):
            rows.append({'date': d,
                         'margin_balance': float(fin.iloc[i] + sec.iloc[i])
                                           if pd.notna(fin.iloc[i]) else None,
                         'margin_fin_balance': float(fin.iloc[i]),
                         'margin_sec_balance': float(sec.iloc[i]),
                         'margin_fin_buy': float(fbuy.iloc[i])})
        upsert(conn, {'margin_balance': None, 'margin_fin_balance': None,
                      'margin_sec_balance': None, 'margin_fin_buy': None},
               rows, now)
        print(f'  两融: {len(rows)} 行 ({df["日期"].iloc[0]} → {df["日期"].iloc[-1]})')

    # ── 2. 股指期货升贴水 ────────────────────────────────────────────────
    if a.only in (None, 'basis'):
        print('=== 2. 股指期货升贴水 ===')
        spot = {}
        try:
            # index_daily 在 stock_daily.db（ingest_index 落库目标），不在 meta 库
            spot_conn = sqlite3.connect(DATABASE_PATH, timeout=60.0)
            spot_df = pd.read_sql_query(
                'SELECT code, date, close FROM index_daily '
                "WHERE code IN ('sh.000300','sh.000016','sh.000905') "
                'ORDER BY date ASC', spot_conn)
            spot_conn.close()
            for code, g in spot_df.groupby('code'):
                spot[code] = g.set_index('date')['close'].astype(float)
        except Exception as e:
            probe_abort(f'读取 index_daily 现货失败: {e}')
        if not spot:
            probe_abort('index_daily 表为空 —— 先跑 update_daily_data.py --only-index-macro')

        pairs = [('IF0', 'sh.000300', 'basis_if'),
                 ('IH0', 'sh.000016', 'basis_ih'),
                 ('IC0', 'sh.000905', 'basis_ic')]
        total_rows = 0
        for sym, spot_code, col in pairs:
            fdf = ak.futures_main_sina(symbol=sym, start_date='20150101',
                                       end_date=datetime.now().strftime('%Y%m%d'))
            if fdf is None or len(fdf) == 0:
                probe_abort(f'期指 {sym} 探针空返回', fdf)
            need = ['日期', '收盘价']
            missing = [c for c in need if c not in fdf.columns]
            if missing:
                probe_abort(f'期指 {sym} 缺列 {missing}', fdf)
            fdf['日期'] = pd.to_datetime(fdf['日期']).dt.strftime('%Y-%m-%d')
            fclose = fdf.set_index('日期')['收盘价'].astype(float)
            sclose = spot[spot_code]
            basis = (fclose / sclose.reindex(fclose.index) - 1.0)
            rows = [{'date': d, col: float(v)}
                    for d, v in basis.items() if pd.notna(v)]
            if rows:
                upsert(conn, {col: None}, rows, now)
            total_rows += len(rows)
            span = (rows[0]['date'], rows[-1]['date']) if rows else ('-', '-')
            print(f'  {sym}→{col}: {len(rows)} 行  {span[0]} → {span[1]}')
        print(f'  期指升贴水合计 {total_rows} 行')

    # ── 3. 宏观利率 ──────────────────────────────────────────────────────
    if a.only in (None, 'rates'):
        print('=== 3. 宏观利率 ===')
        # 3a. SHIBOR
        df = ak.macro_china_shibor_all()
        if df is None or len(df) == 0:
            probe_abort('SHIBOR 探针空返回', df)
        need = ['日期', 'O/N-定价', '1W-定价', '3M-定价']
        missing = [c for c in need if c not in df.columns]
        if missing:
            probe_abort(f'SHIBOR 缺列 {missing}', df)
        df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
        rows = [{'date': r[0],
                 'shibor_on': float(pd.to_numeric(r[1], errors='coerce')),
                 'shibor_1w': float(pd.to_numeric(r[2], errors='coerce')),
                 'shibor_3m': float(pd.to_numeric(r[3], errors='coerce'))}
                for r in df[['日期', 'O/N-定价', '1W-定价', '3M-定价']]
                .itertuples(index=False, name=None)]
        upsert(conn, {'shibor_on': None, 'shibor_1w': None, 'shibor_3m': None},
               rows, now)
        print(f'  SHIBOR: {len(rows)} 行 ({df["日期"].iloc[0]} → {df["日期"].iloc[-1]})')

        # 3b. LPR（月度；TRADE_DATE 为发布日）
        df = ak.macro_china_lpr()
        if df is None or len(df) == 0:
            probe_abort('LPR 探针空返回', df)
        need = ['TRADE_DATE', 'LPR1Y']
        missing = [c for c in need if c not in df.columns]
        if missing:
            probe_abort(f'LPR 缺列 {missing}', df)
        df['日期'] = pd.to_datetime(df['TRADE_DATE']).dt.strftime('%Y-%m-%d')
        rows = [{'date': r[0],
                 'lpr_1y': float(pd.to_numeric(r[1], errors='coerce'))}
                for r in df[['日期', 'LPR1Y']]
                .itertuples(index=False, name=None)]
        upsert(conn, {'lpr_1y': None}, rows, now)
        print(f'  LPR: {len(rows)} 行 ({df["日期"].iloc[0]} → {df["日期"].iloc[-1]})')

        # 3c. 中国国债 10Y 收益率
        df = ak.bond_zh_us_rate(start_date='20100101')
        if df is None or len(df) == 0:
            probe_abort('债券收益率探针空返回', df)
        need = ['日期', '中国国债收益率10年']
        missing = [c for c in need if c not in df.columns]
        if missing:
            probe_abort(f'债券收益率缺列 {missing}', df)
        df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
        rows = [{'date': r[0],
                 'cn10y': float(pd.to_numeric(r[1], errors='coerce'))}
                for r in df[['日期', '中国国债收益率10年']]
                .itertuples(index=False, name=None)]
        upsert(conn, {'cn10y': None}, rows, now)
        print(f'  国债10Y: {len(rows)} 行 ({df["日期"].iloc[0]} → {df["日期"].iloc[-1]})')

    # ── 汇总 ─────────────────────────────────────────────────────────────
    chk = conn.execute('SELECT COUNT(*), MIN(date), MAX(date) '
                       'FROM market_macro_daily').fetchone()
    print(f'\nmarket_macro_daily 合计: {chk[0]} 行  {chk[1]} → {chk[2]}')
    cols = [r[1] for r in conn.execute(
        'PRAGMA table_info(market_macro_daily)')]
    print(f'列: {cols}')
    conn.close()


if __name__ == '__main__':
    main()
