#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段0 市场级数据源探针 v2（只读，不落库）
========================================

在写落库脚本之前，验证四类市场级数据的可用性：
  1. 北向资金净流入（沪深港通）—— stock_hsgt_hist_em / stock_hsgt_fund_flow_summary_em
  2. 融资融券余额（两融）—— stock_margin_sse / stock_margin_szse
  3. 股指期货升贴水（IF/IH/IC/IM 主力 vs 现货指数）—— futures_main_sina
  4. 宏观利率（SHIBOR / LPR / 国债收益率）—— macro_china_shibor_all / macro_china_lpr / bond_zh_us_rate

每类输出：接口名、返回行数、历史起点/终点、列名、日期粒度、PIT 可用性判断。

关键背景（2026-08 现状）：
- 北向资金日度净流入自 **2024-08-19** 起停披露（沪深港通披露机制调整，
  改为仅季度披露持股）。历史数据应可查到 2024-08-16。这直接影响：
  熊市主判窗口 2022-09→2024-08 可用，牛市窗口 2024-08→2026-08 只有前 3 天。
- 两融余额日频持续披露，2010-03 起。
- 股指期货：IF 2010-04 上市，IH/IC 2015-04，IM 2022-07。升贴水 = 主力收盘 - 现货 / 现货。
- 宏观利率：SHIBOR 2006-10 起；LPR 2019-08 起；国债收益率 2002 起。

用法:
  python scripts/exp/probe_phase0_sources.py [--quick]
"""
import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def probe(name, fn):
    print(f"\n{'='*70}\n[{name}]")
    try:
        df = fn()
        if df is None or len(df) == 0:
            print("  -> 空返回")
            return None
        cols = list(df.columns)
        print(f"  行数: {len(df)}  列: {cols[:12]}{'...' if len(cols) > 12 else ''}")
        # 尝试识别日期列
        date_col = next((c for c in cols if 'date' in c.lower() or 'time' in c.lower()
                         or c in ('日期', '时间', '交易日')), None)
        if date_col:
            s = df[date_col].astype(str)
            print(f"  日期列[{date_col}]: {s.iloc[0]} → {s.iloc[-1]}")
        return df
    except Exception as e:
        import traceback
        print(f"  -> 探针异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='只跑北向+两融两类核心')
    args = ap.parse_args()

    import akshare as ak

    # ── 1. 北向资金 ──────────────────────────────────────────────────────
    probe('北向资金·历史净流入(stock_hsgt_hist_em symbol=北向资金)',
          lambda: ak.stock_hsgt_hist_em(symbol="北向资金"))
    probe('北向资金·当日资金流向(stock_hsgt_fund_flow_summary_em)',
          lambda: ak.stock_hsgt_fund_flow_summary_em())

    if args.quick:
        return

    # ── 2. 融资融券 ──────────────────────────────────────────────────────
    probe('两融·上交所余额(stock_margin_sse 2020-01-01~2024-08-16)',
          lambda: ak.stock_margin_sse(start_date="20200101", end_date="20240816"))
    probe('两融·深交所余额(stock_margin_szse 2020-01-01~2024-08-16)',
          lambda: ak.stock_margin_szse(start_date="20200101", end_date="20240816"))

    # ── 3. 股指期货升贴水 ────────────────────────────────────────────────
    for sym, desc in [("IF0", "沪深300期指主力"), ("IH0", "上证50期指主力"),
                      ("IC0", "中证500期指主力"), ("IM0", "中证1000期指主力")]:
        probe(f'期指·{desc}({sym}) futures_main_sina 2022-01-01~2024-08-16',
              lambda s=sym: ak.futures_main_sina(symbol=s, start_date="20220101", end_date="20240816"))

    # ── 4. 宏观利率 ──────────────────────────────────────────────────────
    probe('宏观·SHIBOR 全历史(macro_china_shibor_all)',
          lambda: ak.macro_china_shibor_all())
    probe('宏观·LPR(macro_china_lpr)',
          lambda: ak.macro_china_lpr())
    probe('宏观·中美债收益率(bond_zh_us_rate 2015-01-01~)',
          lambda: ak.bond_zh_us_rate(start_date="20150101"))

    print(f"\n{'='*70}\n探针完成")


if __name__ == '__main__':
    main()
