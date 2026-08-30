"""
个股级市场敏感度特征族（``msens_*``，market sensitivity family）
================================================================

为什么需要这一族
----------------
``idx_*``（T115 晋级的生产实现）已经证明：**把市场信息转成「个股级、随时间变化」
的敏感度特征，是绕过「市场级列零横截面判别力」死结的正确路线**——市场级列每天给
所有股票加同一个常数，在加性 NAM 里零信号；但「个股对市场变量的滚动回归系数」
逐股不同，天然带横截面判别力。

本模块把 ``IndexRelativeFactors`` 的「个股收益 × 指数收益」单变量滚动回归，扩展到
**多个市场变量**（指数收益/指数波动/广度/两融流入/期指升贴水/SHIBOR/国债），产出
一个完整的敏感度特征族。每个 (市场变量 × 滚动窗口) 给出 5 个统计量：

    beta     斜率：个股收益对市场变量变化的敏感度（核心）
    alpha    截距：剔除市场解释后的个股超额（近似个股 alpha）
    r2       拟合优度：个股收益被该市场变量解释的比例
    idio_vol 残差波动：未被市场变量解释的特异风险
    corr     相关系数：个股收益与市场变量的同步性

设计要点（与 ``index_relative_factors.py`` 逐条对齐）
----------------------------------------------------
1. **PIT 安全**：全部是滚动窗口内的历史收益/变化，无未来信息；指数当日收益与
   个股当日收益同为 T 日收盘后可得。
2. **窗口无关**：个股收益序列取自该股在 ``daily_data`` 的全量历史（不是传入的
   行情窗口），多变量滚动在自身日历上算完再 reindex 到目标日期并 ffill，避免
   增量更新时头部退化成热身中性值而产生漂移。
3. **中性填充**：热身期/缺失的中性值。β→1.0（等市场暴露），其余→0.0。
4. **进程内缓存**：指数收益与宏观/广度序列每个子进程只加载一次。

市场变量注册表 ``SENS_REGISTRY``
--------------------------------
每个条目：``(输出前缀, 来源类型, 源列/键, 是否差分)``。
- 差分变量（两融/利率/国债）：先取一阶差分再回归个股收益，捕捉「变化」的敏感度。
- 水平变量（指数收益/指数波动/广度/升贴水）：直接用水平值回归。
- 指数收益需按代码板块映射（60/68→上证，00→深成，30→创业板，其余→上证）。

生产列集 ≠ 全部统计量（重要）
----------------------------
滚动回归本身产出 5 个统计量，但**进缓存的只有 corr/r2 及其时序形态派生**，共
90 列 = 9 变量 × (4 个 level + 6 个 morph)：

    level（36）  msens_{var}_corr_20 / _corr_60 / _r2_20 / _r2_60
    morph（54）  msens_{var}_slope5_corr / _slope10_corr / _slope5_r2
                 msens_{var}_stab20_corr / _change_corr / _change_r2

**beta/alpha/idio_vol 被刻意剔除**：前序诊断（diag_beta_family_ic / diag_beta_morph
+ 两轮冗余校验）证明它们是 vol/反转的代理，与基线 246 列冗余；而 22/54 个形态量
显著（|t|>2 且 |IC7|>0.01）、52/54 与基线低相关（<0.4），是基线未覆盖的真实增量。
巧合提醒：``9×2×5 = 90`` 与 ``9×(4+6) = 90`` 数字相同但**列集完全不同**，
不要靠列数判断口径是否对。

双库：本族的源数据横跨两个库
----------------------------
``index_daily`` 在 ``stock_daily.db``，而 ``market_macro_daily`` /
``market_sentiment`` 在 ``stock_meta.db``。``calculate`` 接口只收一个 ``db_path``
（对齐 ``IndexRelativeFactors``），meta 库由 ``_meta_db_of`` 从同目录推导 ——
这与 ``MarketSentimentFetcher`` 的做法一致，调用方不必知道两个库。

用法
----
    生产（缓存契约 90 列）：``MarketSensitivityFactors.calculate(code, dates, db_path)``
    诊断（5 统计量裸值）：``compute_sensitivity(r_i, x, windows)`` / ``_compute_all``
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.factors.rolling_stats import align_to_dates, rolling_slope

# ── 滚动窗口与热身 ──────────────────────────────────────────────────────────
WINDOWS: List[int] = [20, 60]
MINP_RATIO = 0.66  # 最小样本数 = window * 0.66

# 每个统计量对应的中性填充值
NEUTRAL = {
    'beta': 1.0,
    'alpha': 0.0,
    'r2': 0.0,
    'idio_vol': 0.0,
    'corr': 0.0,
}

STATS = ['beta', 'alpha', 'r2', 'idio_vol', 'corr']

# 输出列名模板：``{prefix}_{stat}_{window}``
# 例：``msens_idxret_beta_60``、``msens_marginflow_alpha_20``

# 市场变量注册表
#   kind:
#     'index_ret'  → 指数日收益，需按代码板块映射（level 回归）
#     'index_vol'  → 指数收益 20 日滚动 std（全局序列，level 回归）
#     'global'     → 单一全局市场序列（来自 load_market_series）
#   diff: 是否先对源序列取一阶差分
SENS_REGISTRY: List[Dict] = [
    {'prefix': 'msens_idxret',      'kind': 'index_ret', 'src': None,          'diff': False},
    {'prefix': 'msens_idxvol',      'kind': 'index_vol', 'src': None,          'diff': False},
    {'prefix': 'msens_breadth_up',  'kind': 'global',    'src': 'breadth_up',  'diff': False},
    {'prefix': 'msens_breadth_ma',  'kind': 'global',    'src': 'breadth_ma20','diff': False},
    {'prefix': 'msens_marginflow',  'kind': 'global',    'src': 'margin_flow','diff': True},
    {'prefix': 'msens_finbuyflow',  'kind': 'global',    'src': 'margin_finbuy_flow','diff': True},
    {'prefix': 'msens_basis_if',    'kind': 'global',    'src': 'basis_if',    'diff': False},
    {'prefix': 'msens_shibor3m',    'kind': 'global',    'src': 'shibor3m_chg','diff': True},
    {'prefix': 'msens_cn10y',       'kind': 'global',    'src': 'cn10y_chg',   'diff': True},
]

MARKET_INDEX = 'sh.000300'

# ── 生产缓存契约（90 列）─────────────────────────────────────────────────────
# 只有 corr/r2 的 level 与形态派生进缓存；beta/alpha/idio_vol 不进（与基线冗余）。
USE_STATS: Tuple[str, ...] = ('corr', 'r2')
LEVEL_WINDOWS: Tuple[int, ...] = (20, 60)
# 形态派生种类（列名后缀，顺序即列序）
MORPH_KINDS: Tuple[str, ...] = ('slope5_corr', 'slope10_corr', 'slope5_r2',
                                'stab20_corr', 'change_corr', 'change_r2')


def market_sensitivity_columns() -> List[str]:
    """90 个生产列名（确定性顺序：每变量 level 4 列 + morph 6 列）。"""
    cols: List[str] = []
    for entry in SENS_REGISTRY:
        p = entry['prefix']
        for w in LEVEL_WINDOWS:
            for st in USE_STATS:
                cols.append(f'{p}_{st}_{w}')
        for kind in MORPH_KINDS:
            cols.append(f'{p}_{kind}')
    return cols


MSENS_COLUMNS: List[str] = market_sensitivity_columns()

# 中性填充：corr/r2 与全部形态量的中性值都是 0.0（不存在 beta 那种 1.0 中性量，
# 因为 beta 根本不进生产列集）。
MSENS_FILLS: Dict[str, float] = {c: 0.0 for c in MSENS_COLUMNS}

# 进程内缓存（键=db 绝对路径）
_INDEX_CACHE: Dict[str, Dict[str, pd.Series]] = {}
_GLOBAL_CACHE: Dict[str, Dict[str, pd.Series]] = {}


def _meta_db_of(db_path: str) -> str:
    """从行情库路径推导 meta 库路径（两者同目录），对齐 MarketSentimentFetcher。"""
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), 'stock_meta.db')


# ════════════════════════════════════════════════════════════════════════════
# 数据加载（进程内缓存）
# ════════════════════════════════════════════════════════════════════════════
def load_index_returns(db_path: str) -> Optional[Dict[str, pd.Series]]:
    """读指数日收益（date 字符串索引）。表缺失返回 None。"""
    key = os.path.abspath(db_path)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key] or None
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        if not _table_exists(conn, 'index_daily'):
            _INDEX_CACHE[key] = {}
            return None
        df = pd.read_sql_query(
            'SELECT code, date, pctChg FROM index_daily ORDER BY date ASC', conn)
    finally:
        conn.close()
    if df.empty:
        _INDEX_CACHE[key] = {}
        return None
    out: Dict[str, pd.Series] = {}
    for code, g in df.groupby('code'):
        out[code] = pd.to_numeric(
            g.set_index('date')['pctChg'], errors='coerce') / 100.0
    if 'sz.399006' in out and 'sz.399001' in out:
        gem = out['sz.399006'].reindex(out['sz.399001'].index)
        out['sz.399006_padded'] = gem.fillna(out['sz.399001'])
    if MARKET_INDEX not in out:
        _INDEX_CACHE[key] = {}
        return None
    _INDEX_CACHE[key] = out
    return out


def load_market_series(db_path: str) -> Optional[Dict[str, pd.Series]]:
    """读宏观/广度市场序列（date 字符串索引），并派生差分/波动率序列。

    返回 dict：breadth_up / breadth_ma20 / margin_flow / margin_finbuy_flow /
    basis_if / shibor3m_chg / cn10y_chg（均为全局市场级序列）。
    """
    key = os.path.abspath(db_path)
    if key in _GLOBAL_CACHE:
        return _GLOBAL_CACHE[key] or None
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        has_macro = _table_exists(conn, 'market_macro_daily')
        has_sent = _table_exists(conn, 'market_sentiment')
        macro = pd.read_sql_query(
            'SELECT date, margin_balance, margin_fin_buy, basis_if, shibor_3m, cn10y '
            'FROM market_macro_daily ORDER BY date ASC', conn) if has_macro else pd.DataFrame()
        sent = pd.read_sql_query(
            'SELECT date, up_ratio, breadth_ma20 FROM market_sentiment '
            'ORDER BY date ASC', conn) if has_sent else pd.DataFrame()
    finally:
        conn.close()
    if macro.empty and sent.empty:
        _GLOBAL_CACHE[key] = {}
        return None

    out: Dict[str, pd.Series] = {}
    if not macro.empty:
        macro['date'] = macro['date'].astype(str)
        macro = macro.set_index('date')
        out['margin_flow'] = macro['margin_balance'].ffill().diff()
        out['margin_finbuy_flow'] = macro['margin_fin_buy'].ffill().diff()
        out['basis_if'] = macro['basis_if'].ffill()
        out['shibor3m_chg'] = macro['shibor_3m'].ffill().diff()
        out['cn10y_chg'] = macro['cn10y'].ffill().diff()
    if not sent.empty:
        sent['date'] = sent['date'].astype(str)
        sent = sent.set_index('date')
        out['breadth_up'] = sent['up_ratio'].ffill()
        out['breadth_ma20'] = sent['breadth_ma20'].ffill()

    out = {k: v.astype(np.float64) for k, v in out.items() if len(v) > 0}
    _GLOBAL_CACHE[key] = out
    return out


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def board_series(code: str, idx: Dict[str, pd.Series]) -> pd.Series:
    """板块基准映射（不带前缀的 6 位代码）。"""
    if code.startswith(('60', '68')):
        return idx['sh.000001']
    if code.startswith('30'):
        return idx.get('sz.399006_padded', idx['sz.399001'])
    if code.startswith('00'):
        return idx['sz.399001']
    return idx['sh.000001']


# ════════════════════════════════════════════════════════════════════════════
# 核心：单变量滚动回归
# ════════════════════════════════════════════════════════════════════════════
def compute_sensitivity(r_i: pd.Series, x: pd.Series,
                        windows: Sequence[int] = WINDOWS) -> pd.DataFrame:
    """个股收益 ``r_i`` 对市场变量 ``x`` 的滚动回归。

    两端均以 date 字符串为索引，且已对齐到同一交易日历。
    市场变量缺值的日期不参与滚动（min_periods 兜底）。

    返回与 ``r_i`` 等索引的 DataFrame，列 ``{prefix}_{stat}_{w}`` 由调用方拼。
    这里只返回裸值，列名在外面统一加前缀。
    """
    df = pd.DataFrame({'r': r_i})
    df['x'] = x.reindex(df.index)
    out = pd.DataFrame(index=df.index)
    for w in windows:
        mp = max(3, int(round(w * MINP_RATIO)))
        r = df['r']
        xx = df['x']
        cov_rx = r.rolling(w, min_periods=mp).cov(xx)
        var_x = xx.rolling(w, min_periods=mp).var()
        var_r = r.rolling(w, min_periods=mp).var()
        mean_r = r.rolling(w, min_periods=mp).mean()
        mean_x = xx.rolling(w, min_periods=mp).mean()
        with np.errstate(invalid='ignore', divide='ignore'):
            beta = cov_rx / var_x.replace(0.0, np.nan)
            denom = np.sqrt(var_r * var_x.replace(0.0, np.nan))
            corr = cov_rx / denom.replace(0.0, np.nan)
        alpha = mean_r - beta * mean_x
        resid = r - mean_r + beta * (mean_x - xx)  # = r - alpha - beta*x
        idio = resid.rolling(w, min_periods=mp).std()
        r2 = corr * corr
        out[f'beta_{w}'] = beta
        out[f'alpha_{w}'] = alpha
        out[f'r2_{w}'] = r2
        out[f'idio_vol_{w}'] = idio
        out[f'corr_{w}'] = corr
    return out.replace([np.inf, -np.inf], np.nan)


# ════════════════════════════════════════════════════════════════════════════
# 生产列集：corr/r2 的 level + 时序形态派生（90 列）
# ════════════════════════════════════════════════════════════════════════════
def derive_morph(corr20: pd.Series, corr60: pd.Series,
                 r2_20: pd.Series, r2_60: pd.Series,
                 prefix: str) -> Dict[str, pd.Series]:
    """从单股 corr/r2 序列派生 6 个时序形态量。

    形态量必须在**全历史** corr/r2 序列上算完再 reindex 到目标日期——先 reindex
    再求斜率会让停牌日参与窗口，与已物化的缓存值对不上。
    """
    return {
        f'{prefix}_slope5_corr': rolling_slope(corr20, 5),
        f'{prefix}_slope10_corr': rolling_slope(corr20, 10),
        f'{prefix}_slope5_r2': rolling_slope(r2_20, 5),
        f'{prefix}_stab20_corr': corr20.rolling(20, min_periods=10).std(),
        f'{prefix}_change_corr': corr20 - corr60,
        f'{prefix}_change_r2': r2_20 - r2_60,
    }


def compute_production_features(r_i: pd.Series, code: str,
                                idx: Optional[Dict[str, pd.Series]],
                                idx_vol: Optional[pd.Series],
                                glob: Optional[Dict[str, pd.Series]]) -> pd.DataFrame:
    """单股全历史的 90 列生产特征（date 字符串索引，**未** reindex）。

    先用 ``_compute_all`` 拿全部 5 统计量，再挑 corr/r2 并在其上派生形态量。
    某个市场变量的源序列缺失时该变量的 10 列整体缺席，由 ``align_to_dates``
    按中性 0.0 补齐（与原离线注入口径一致）。
    """
    raw = _compute_all(r_i, code, idx, idx_vol, glob)
    if raw.empty or raw.columns.empty:
        return pd.DataFrame(index=r_i.index)

    parts: List[pd.DataFrame] = []
    for entry in SENS_REGISTRY:
        p = entry['prefix']
        need = {f'{st}_{w}': f'{p}_{st}_{w}'
                for st in USE_STATS for w in LEVEL_WINDOWS}
        if not all(v in raw.columns for v in need.values()):
            continue
        corr20 = raw[need['corr_20']]
        corr60 = raw[need['corr_60']]
        r2_20 = raw[need['r2_20']]
        r2_60 = raw[need['r2_60']]
        level = pd.DataFrame({
            f'{p}_corr_20': corr20, f'{p}_corr_60': corr60,
            f'{p}_r2_20': r2_20, f'{p}_r2_60': r2_60,
        }, index=raw.index)
        morph = pd.DataFrame(derive_morph(corr20, corr60, r2_20, r2_60, p),
                             index=raw.index)
        parts.append(pd.concat([level, morph], axis=1))

    if not parts:
        return pd.DataFrame(index=r_i.index)
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def load_stock_returns(code: str, db_path: str) -> pd.Series:
    """该股在 ``daily_data`` 的全量日收益（date 字符串索引）。空表返回空 Series。"""
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        d = pd.read_sql_query(
            'SELECT date, pctChg FROM daily_data WHERE code=? ORDER BY date ASC',
            conn, params=(code,))
    finally:
        conn.close()
    if d.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(d.set_index('date')['pctChg'], errors='coerce') / 100.0


def index_volatility(idx: Optional[Dict[str, pd.Series]]) -> Optional[pd.Series]:
    """沪深300 收益的 20 日滚动 std（``msens_idxvol`` 的源序列）。"""
    if not idx:
        return None
    return idx[MARKET_INDEX].rolling(20, min_periods=13).std()


# ════════════════════════════════════════════════════════════════════════════
# 对外：单股全族计算（生产物化用）
# ════════════════════════════════════════════════════════════════════════════
class MarketSensitivityFactors:
    """个股级市场敏感度特征计算器，由综合计算器持有，风格对齐 IndexRelativeFactors。"""

    @staticmethod
    def calculate(code: str, dates: Sequence, db_path: str) -> pd.DataFrame:
        """返回与 ``dates`` 等长的 **90 列** DataFrame（float32，无 NaN）。

        个股收益取自 daily_data 全量历史（窗口无关）；市场序列跨两库加载
        （index_daily 在 ``db_path``，宏观/广度在同目录 ``stock_meta.db``）；
        形态派生在全历史上算完后再 reindex 到 ``dates`` 并 ffill + 中性填充。

        指数与宏观序列**全都**不可用时返回空 DataFrame —— 让整族列缺席去撞
        下游的列完整性护栏，而不是造 90 根常数 0 假列（那会无声退化成噪声）。
        """
        if len(dates) == 0:
            return pd.DataFrame()
        idx = load_index_returns(db_path)
        glob = load_market_series(_meta_db_of(db_path))
        if idx is None and glob is None:
            return pd.DataFrame()

        r_i = load_stock_returns(code, db_path)
        feats = pd.DataFrame() if r_i.empty else compute_production_features(
            r_i, code, idx, index_volatility(idx), glob)
        return align_to_dates(feats, dates, MSENS_COLUMNS, MSENS_FILLS)


def _compute_all(r_i: pd.Series, code: str,
                 idx: Optional[Dict[str, pd.Series]],
                 idx_vol: Optional[pd.Series],
                 glob: Optional[Dict[str, pd.Series]]) -> pd.DataFrame:
    """对单股算全族（内部，未 reindex）。"""
    pieces: List[pd.DataFrame] = []
    for entry in SENS_REGISTRY:
        prefix = entry['prefix']
        if entry['kind'] == 'index_ret':
            if idx is None:
                continue
            x = board_series(code, idx)
        elif entry['kind'] == 'index_vol':
            if idx_vol is None:
                continue
            x = idx_vol
        else:  # global
            if glob is None or entry['src'] not in glob:
                continue
            x = glob[entry['src']]
        if entry['diff']:
            x = x.diff()
        sub = compute_sensitivity(r_i, x)
        sub = sub.rename(columns=lambda c: f'{prefix}_{c}')
        pieces.append(sub)
    if not pieces:
        return pd.DataFrame(index=r_i.index, columns=_all_columns())
    return pd.concat(pieces, axis=1)


def _all_columns() -> List[str]:
    cols = []
    for entry in SENS_REGISTRY:
        for w in WINDOWS:
            for s in STATS:
                cols.append(f"{entry['prefix']}_{s}_{w}")
    return cols


def _neutral_of(col: str) -> float:
    """从列名解析统计量并取中性填充值。"""
    for s in STATS:
        if f'_{s}_' in col:
            return NEUTRAL[s]
    return 0.0
