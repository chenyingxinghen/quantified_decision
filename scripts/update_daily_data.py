#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全市场数据增量更新脚本 (Baostock 版)

一次跑完日常所需的四类数据，顺序**有依赖**，不要随意调换：

  1. 个股日频行情      → stock_daily.db
  2. 个股财务基本面    → stock_finance.db
  3. 指数日线 + 货币供应 → index_daily / macro_money_supply
  4. 因子缓存增量更新  → TrainingConfig.CURRENT_CACHE_DIR

第 3 步必须排在第 4 步之前：``index_daily`` 是 5 列 ``idx_*`` 指数相对因子
（idx_beta_60 / idx_corr_20 / idx_idio_vol_20 / idx_rs_20 / idx_rs_60）的原料。
指数没更新到最新交易日的话，因子计算器会让这些列**缺席**（见
comprehensive_factor_calculator 的 index_daily 分支），推理侧的列完整性护栏
随后就会硬失败。

指数/货币供应的落库逻辑原本在独立的 scripts/ingest_index_and_macro.py，
2026-08-22 并入本脚本 —— 它本就是"每日数据更新"的一部分，单独放一个脚本
只会让人忘记跑它。

沿用 E10 预告落库的三条纪律：
  1. 探针先行：接口空返回 = 登录/配额/接口坏，立即中止，不写任何进度；
  2. 字段名硬校验：缺关键列直接 ABORT 并打印实际列名（query_forecast_report
     的 pubDate 改名坑真踩过一次）；
  3. 幂等可重跑：INSERT OR REPLACE + 主键，重跑不产生重复行。

用法:
  python scripts/update_daily_data.py                      # 全市场增量（含指数/货币/因子缓存）
  python scripts/update_daily_data.py --skip-index --skip-money
  python scripts/update_daily_data.py --mode single --symbol sh.600000
  python scripts/update_daily_data.py --probe-constituents # 只探针成分股 API 深度
"""
import sys
import os
import argparse
import sqlite3
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.data.baostock_main import BaostockDataManager
import config
from config import DATABASE_PATH


# ==============================================================================
# 指数日线 + 货币供应落库
# ==============================================================================

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
    """指数日线落库。用途是**逐股截面列**（beta/相对强弱/特异波动），
    不是给门控喂 regime —— 门控轴已终审关闭（T108/T111）。"""
    from core.data.baostock_fetcher_methods import _bs_query

    conn = sqlite3.connect(DATABASE_PATH, timeout=60.0)
    conn.execute(INDEX_DDL)
    conn.execute('CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily (date)')
    conn.commit()

    fields = 'date,code,open,high,low,close,preclose,volume,amount,pctChg'
    ok = True
    try:
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
            conn.executemany(
                'INSERT OR REPLACE INTO index_daily VALUES (?,?,?,?,?,?,?,?,?,?)', rows)
            conn.commit()
            span = (rows[0][1], rows[-1][1]) if rows else ('-', '-')
            print(f'  {code}: {len(rows)} 行  {span[0]} → {span[1]}')
        n = conn.execute('SELECT COUNT(*), COUNT(DISTINCT code) FROM index_daily').fetchone()
        print(f'index_daily 合计: {n[0]} 行 / {n[1]} 个指数')
    finally:
        conn.close()
    return ok


def ingest_money(start_ym: str, end_ym: str) -> bool:
    """货币供应月度落库。M1-M2 剪刀差是标量门终检（T116）唯一的外生条件化信号。

    PIT 对齐一律按 ``statMonth 次月 15 日``才可见（月度数据发布滞后约次月
    10~15 日），特征侧不得直接用 statMonth 当可见日。
    """
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
    try:
        conn.execute('DROP TABLE IF EXISTS macro_money_supply')
        conn.execute(MONEY_DDL)
        conn.executemany(
            'INSERT OR REPLACE INTO macro_money_supply VALUES (?,?,?,?,?,?,?,?,?,?,?)', rows)
        conn.commit()
        chk = conn.execute(
            'SELECT MIN(statMonth), MAX(statMonth), COUNT(*) FROM macro_money_supply').fetchone()
        nn = conn.execute('SELECT COUNT(*) FROM macro_money_supply '
                          'WHERE m1YOY IS NOT NULL AND m2YOY IS NOT NULL').fetchone()[0]
        print(f'macro_money_supply: {chk[2]} 行  {chk[0]} → {chk[1]}  '
              f'(m1YOY&m2YOY 非空 {nn})')
    finally:
        conn.close()
    return True


def probe_constituents():
    """只探针不落库：验证 query_hs300_stocks(date) 的历史深度。

    若只有近期快照，拿它做历史特征就是幸存者偏差 —— 所以先探针，不建表。
    """
    from core.data.baostock_fetcher_methods import _bs_query
    print('成分股 API 历史深度探针（只打印，不建表）:')
    for d in ['2010-06-30', '2015-06-30', '2020-06-30', '2024-06-28', '2026-08-14']:
        for api in ['query_hs300_stocks', 'query_zz500_stocks', 'query_sz50_stocks']:
            rs = _bs_query(api, date=d)
            n = len(rs.data) if rs is not None else -1
            print(f'  {api}({d}): {n} 行')


def update_index_and_macro(start: str, end: str,
                           do_index: bool = True, do_money: bool = True) -> bool:
    """指数 + 货币供应落库。自带 baostock 登录/注销，可独立调用。"""
    from core.data.baostock_fetcher import BaostockFetcher
    if not BaostockFetcher._bs_login():
        print('[ABORT] baostock 登录失败')
        return False
    ok = True
    try:
        if do_index:
            ok &= ingest_index(start, end)
        if do_money:
            ok &= ingest_money(start[:7], end[:7])
    finally:
        _bs_logout()
    return ok


def _bs_logout():
    """注销并释放会话锁。BaostockFetcher.logout 是实例方法但只操作类态。"""
    try:
        import baostock as bs
        from core.data.baostock_fetcher import BaostockFetcher, _release_bs_session_lock
        bs.logout()
        BaostockFetcher._global_bs_logged_in = False
        _release_bs_session_lock()
        print('Baostock 已注销')
    except Exception:
        pass


# ==============================================================================
# 个股行情 / 财务
# ==============================================================================

def update_single_stock(symbol, incremental=False, start_date=None, end_date=None):
    """更新单只股票数据"""
    print(f"\n开始更新股票 {symbol} | 模式: {'增量' if incremental else '全量'}")
    manager = BaostockDataManager()
    try:
        # 单只更新时仍保持原有逻辑顺序
        manager.update_stock_data(symbol, incremental=incremental, start_date=start_date, end_date=end_date)
        manager.update_finance_data(symbol, incremental=incremental)
    finally:
        manager.close()
        manager.logout()

def update_multiple_stocks(symbols, incremental=True, workers=None, start_date=None, end_date=None):
    """批量更新多只股票数据"""
    if workers is None:
        workers = config.WORKERS_NUM

    print(f"\n正在批量更新 {len(symbols)} 只股票 (并发数: {workers}) | 模式: {'增量' if incremental else '全量'}")
    manager = BaostockDataManager()
    try:
        # 核心逻辑：先同步所有股票的日频数据
        print("\n--- 第一步: 同步日频行情数据 ---")
        manager.update_specific_stocks(symbols, incremental=incremental, workers=workers, mode='daily', start_date=start_date, end_date=end_date)

        # 再同步所有股票的财务数据
        print("\n--- 第二步: 同步财务基本面数据 ---")
        manager.update_specific_stocks(symbols, incremental=incremental, workers=workers, mode='finance')
    finally:
        manager.close()

def update_all_stocks(incremental=True, workers=None, start_date=None, end_date="2030-01-01",
                      do_index=True, do_money=True, index_start='2005-01-01'):
    """更新所有股票数据 + 指数/货币供应 + 因子缓存。"""
    if workers is None:
        workers = config.WORKERS_NUM

    print(f"\n=== 开始同步全市场数据 (源: Baostock) | 模式: {'增量' if incremental else '全量'} ===")
    manager = BaostockDataManager()
    try:
        # 第一步：获取日频数据
        print("\n--- 第一步: 同步全市场日频数据 ---")
        manager.init_all_stocks(incremental=incremental, workers=workers, mode='daily', start_date=start_date, end_date=end_date)

        # 第二步：获取财务数据
        print("\n--- 第二步: 同步全市场财务数据 ---")
        manager.init_all_stocks(incremental=incremental, workers=workers, mode='finance')
    finally:
        manager.close()

    # 第三步：指数日线 + 货币供应。
    # 必须在因子缓存之前 —— index_daily 是 5 列 idx_* 的原料，指数没更新到最新
    # 交易日的话这些列会缺席，推理侧的列完整性护栏随后硬失败。
    if do_index or do_money:
        print("\n--- 第三步: 同步指数日线 / 货币供应 ---")
        _end = datetime.now().strftime('%Y-%m-%d')
        if not update_index_and_macro(index_start, _end, do_index=do_index, do_money=do_money):
            print("[警告] 指数/货币供应落库未完全成功 —— idx_* 因子可能缺最新交易日")
    else:
        print("\n--- 第三步: 跳过指数/货币供应 (--skip-index / --skip-money) ---")

    # 第四步：更新因子缓存
    print("\n--- 第四步: 更新选股因子缓存 ---")
    try:
        from scripts.select_stocks import _update_factor_cache_incremental, get_all_stock_codes
        from config.factor_config import TrainingConfig
        # 当前版本化基座（含 idx_* / fc_*），不是无清单的历史共享缓存。
        cache_dir = os.path.join(PROJECT_ROOT, TrainingConfig.CURRENT_CACHE_DIR)
        all_codes = get_all_stock_codes(DATABASE_PATH)
        _update_factor_cache_incremental(
            db_path=DATABASE_PATH,
            codes=all_codes,
            cache_dir=cache_dir,
            workers=workers,
        )
        print("✓ 因子缓存更新完成")
    except Exception as e:
        import traceback
        print(f"[警告] 因子缓存更新失败，不影响行情数据: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='股票数据增量更新脚本')
    parser.add_argument('--mode', choices=['single', 'multiple', 'all'], default='all')
    parser.add_argument('--symbol', type=str, help='股票代码 (mode=single)')
    parser.add_argument('--symbols', type=str, nargs='+', help='代码列表 (mode=multiple)')
    parser.add_argument('--full', action='store_true', help='全量更新 (从 10 年前开始)')
    parser.add_argument('--workers', type=int, default=None, help=f'并发线程数 (默认: {config.WORKERS_NUM})')
    parser.add_argument('--start', type=str, default='2010-01-01', help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2030-01-01', help='结束日期 (YYYY-MM-DD)')

    # ── 指数 / 宏观 ──
    parser.add_argument('--skip-index', action='store_true',
                        help='跳过指数日线落库（idx_* 因子的原料，日常不要跳）')
    parser.add_argument('--skip-money', action='store_true',
                        help='跳过货币供应月度落库')
    parser.add_argument('--index-start', type=str, default='2005-01-01',
                        help='指数/货币供应起始日期，默认 2005-01-01 全量幂等重刷')
    parser.add_argument('--only-index-macro', action='store_true',
                        help='只跑指数+货币供应，不动个股行情与因子缓存')
    parser.add_argument('--probe-constituents', action='store_true',
                        help='只探针成分股 API 的历史深度（不建表不落库），跑完即退出')

    args = parser.parse_args()
    incremental = not args.full

    if args.probe_constituents:
        from core.data.baostock_fetcher import BaostockFetcher
        if not BaostockFetcher._bs_login():
            print('[ABORT] baostock 登录失败')
            return 1
        try:
            probe_constituents()
        finally:
            _bs_logout()
        return 0

    if args.only_index_macro:
        _end = datetime.now().strftime('%Y-%m-%d')
        ok = update_index_and_macro(args.index_start, _end,
                                    do_index=not args.skip_index,
                                    do_money=not args.skip_money)
        return 0 if ok else 1

    if args.mode == 'single' and args.symbol:
        update_single_stock(args.symbol, incremental=incremental, start_date=args.start, end_date=args.end)
    elif args.mode == 'multiple' and args.symbols:
        update_multiple_stocks(args.symbols, incremental=incremental, workers=args.workers, start_date=args.start, end_date=args.end)
    else:
        update_all_stocks(incremental=incremental, workers=args.workers,
                          start_date=args.start, end_date=args.end,
                          do_index=not args.skip_index,
                          do_money=not args.skip_money,
                          index_start=args.index_start)
    return 0

if __name__ == "__main__":
    sys.exit(main())
