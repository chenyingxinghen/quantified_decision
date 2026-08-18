"""T115 第一步：把指数相对特征追加进因子缓存（复制到新 cache-dir，不动旧缓存）。

设计
----
不走 factor_calculator 全量重算（242 列 × 5447 只 × 13 年重算要数小时），而是：
  源缓存 factors_cache_2026-08-14-fwdadjust 逐文件复制 + 按 date 左连接 5 个新列
  → 目标 factors_cache_2026-08-18-idxrel
通过 T115 判定后再把公式集成进 factor_calculator（生产路径）；未通过则整个目录
作废，旧缓存零污染。这也符合「新公式训练必须显式 --cache-dir」的缓存契约。

5 个新列（统一 ``idx_`` 前缀 → 分组 index_rel，见 config/factor_groups.py）
--------------------------------------------------------------------------
  idx_beta_60     对沪深300 的 60 日滚动 β（cov/var，min_periods=40）
  idx_corr_20     与沪深300 的 20 日滚动相关（min_periods=15）
  idx_idio_vol_20 特异波动：r_s − β·r_m 残差的 20 日 std（β 用同日滚动值）
  idx_rs_20/60    对**板块指数**的相对强弱：Σ20/60日(r_s − r_board)

口径
----
- 收益一律用 pctChg/100（交易所口径，preclose 已含分红除权 → 免复权）。
- 板块映射：60/68 → sh.000001；00 → sz.399001；30 → sz.399006
  （创业板指 2010-06 才有，此前回退 sz.399001）；北交所(4/8/9) → sh.000001。
- 缺失策略与现缓存一致（缓存不落 NaN）：逐股 ffill 后
  β→1.0、corr→0.0、rs→0.0、idio_vol→0.0（仅影响每只票头 ~3 个月热身期）。

用法:
  python scripts/build_idxrel_cache.py [--limit 20]   # limit 用于冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, 'database', 'system_data', 'factors_cache_2026-08-14-fwdadjust')
DST = os.path.join(ROOT, 'database', 'system_data', 'factors_cache_2026-08-18-idxrel')
DAILY_DB = os.path.join(ROOT, 'database', 'stock_daily.db')

NEW_COLS = ['idx_beta_60', 'idx_corr_20', 'idx_idio_vol_20', 'idx_rs_20', 'idx_rs_60']
FILLS = {'idx_beta_60': 1.0, 'idx_corr_20': 0.0, 'idx_idio_vol_20': 0.0,
         'idx_rs_20': 0.0, 'idx_rs_60': 0.0}


def load_index_returns() -> dict:
    conn = sqlite3.connect(DAILY_DB, timeout=60.0)
    df = pd.read_sql_query(
        "SELECT code, date, pctChg FROM index_daily ORDER BY date ASC", conn)
    conn.close()
    out = {}
    for code, g in df.groupby('code'):
        s = pd.to_numeric(g.set_index('date')['pctChg'], errors='coerce') / 100.0
        out[code] = s
    # 板块指数：创业板指成立(2010-06)之前回退深成指
    gem = out['sz.399006'].reindex(out['sz.399001'].index)
    out['sz.399006_padded'] = gem.fillna(out['sz.399001'])
    return out


def board_series(code: str, idx: dict) -> pd.Series:
    if code.startswith(('60', '68')):
        return idx['sh.000001']
    if code.startswith('30'):
        return idx['sz.399006_padded']
    if code.startswith('00'):
        return idx['sz.399001']
    return idx['sh.000001']


def compute_features(r_s: pd.Series, r_m: pd.Series, r_b: pd.Series) -> pd.DataFrame:
    """r_* 均以 date 字符串为索引；在个股自身交易日历上计算。"""
    df = pd.DataFrame({'rs': r_s})
    df['rm'] = r_m.reindex(df.index)
    df['rb'] = r_b.reindex(df.index)
    # 停牌/指数缺失日：市场收益缺失时该日不参与统计（rolling min_periods 兜底）
    cov = df['rs'].rolling(60, min_periods=40).cov(df['rm'])
    var = df['rm'].rolling(60, min_periods=40).var()
    beta = cov / var.replace(0.0, np.nan)
    resid = df['rs'] - beta * df['rm']
    out = pd.DataFrame(index=df.index)
    out['idx_beta_60'] = beta
    out['idx_corr_20'] = df['rs'].rolling(20, min_periods=15).corr(df['rm'])
    out['idx_idio_vol_20'] = resid.rolling(20, min_periods=15).std()
    rel = df['rs'] - df['rb']
    out['idx_rs_20'] = rel.rolling(20, min_periods=15).sum()
    out['idx_rs_60'] = rel.rolling(60, min_periods=40).sum()
    return out.replace([np.inf, -np.inf], np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    os.makedirs(DST, exist_ok=True)
    idx = load_index_returns()
    print(f'指数收益序列: {sorted(k for k in idx if not k.endswith("_padded"))}')

    files = sorted(f for f in os.listdir(SRC) if f.endswith('_factors.parquet'))
    if a.limit:
        files = files[:a.limit]
    todo = [f for f in files if not os.path.exists(os.path.join(DST, f))]
    print(f'共 {len(files)} 文件，已存在 {len(files)-len(todo)}，待处理 {len(todo)}')

    hs300 = idx['sh.000300']

    def one(fname: str) -> int:
        code = fname.split('_')[0]
        src_df = pd.read_parquet(os.path.join(SRC, fname))
        if 'date' not in src_df.columns:
            # 无日期列无法对齐——按缺失策略填充中性值，并计数上报
            for c, v in FILLS.items():
                src_df[c] = v
            src_df.to_parquet(os.path.join(DST, fname), index=False)
            return -1
        conn = sqlite3.connect(DAILY_DB, timeout=60.0)
        d = pd.read_sql_query(
            'SELECT date, pctChg FROM daily_data WHERE code=? ORDER BY date ASC',
            conn, params=(code,))
        conn.close()
        dates = src_df['date'].astype(str)
        if d.empty:
            feats = pd.DataFrame(index=dates.values, columns=NEW_COLS, dtype=float)
        else:
            r_s = pd.to_numeric(d.set_index('date')['pctChg'], errors='coerce') / 100.0
            feats = compute_features(r_s, hs300, board_series(code, idx))
        feats = feats.reindex(dates.values).ffill()
        for c, v in FILLS.items():
            src_df[c] = pd.to_numeric(feats[c], errors='coerce').fillna(v).astype(np.float32).values
        src_df.to_parquet(os.path.join(DST, fname), index=False)
        return len(src_df)

    from joblib import Parallel, delayed
    t0 = time.time()
    results = []
    B = 500
    for i in range(0, len(todo), B):
        chunk = todo[i:i + B]
        results += Parallel(n_jobs=a.workers)(delayed(one)(f) for f in chunk)
        el = time.time() - t0
        done_n = i + len(chunk)
        print(f'  {done_n}/{len(todo)}  {el:.0f}s  (预计还需 {el/max(done_n,1)*(len(todo)-done_n):.0f}s)')

    no_date = sum(1 for r in results if r == -1)
    # 缓存版本契约（core/factors/cache_manifest.py）：显式 --cache-dir 目录里有
    # parquet 但没有 factor_cache_manifest.json 时，训练器硬失败。这里用官方
    # helper 写清单 —— 版本号仍是配置里的那个，因为**基础 242 列的代码公式没变**
    # （逐列 equals 已验证），新增 5 列是本脚本注入的实验列，不由 factor_calculator
    # 产生。把这件事写进清单的 note，避免以后有人把它当成代码生成的缓存。
    from core.factors.cache_manifest import manifest_path, write_cache_manifest
    write_cache_manifest(DST)
    with open(manifest_path(DST), encoding='utf-8') as f:
        _mf = json.load(f)
    _mf['injected_columns'] = NEW_COLS
    _mf['injected_by'] = 'scripts/build_idxrel_cache.py'
    _mf['base_cache'] = os.path.basename(SRC)
    _mf['note'] = ('T115 实验缓存：基础列逐列复制自 base_cache（已验证 equals），'
                   '另注入 5 列指数相对特征。通过判定后应把公式集成进 '
                   'factor_calculator 并 bump FACTOR_DEFINITION_VERSION 全量重建。')
    with open(manifest_path(DST), 'w', encoding='utf-8') as f:
        json.dump(_mf, f, ensure_ascii=False, indent=2)
    print(f'完成: {len(results)} 文件（无 date 列 {no_date} 个），耗时 {time.time()-t0:.0f}s')
    print(f'目标目录: {DST}')


if __name__ == '__main__':
    main()
