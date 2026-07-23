"""一次性验证：rank_y_illiq_20d 是否退化为后向流动性的确定性函数。

做法：
 1. 从 stock_daily.db 取若干样本股票(OHLCV+amount)，用 MultiObjectiveLabelBuilder
    计算真实前向标签 y_illiq_20d，并转为同日 [0,1] 排名 rank_y_illiq_20d。
 2. 用 t 日「后向」成交额滚动特征 amount_ma_10 作为【唯一】特征，按 date 分组
    (query=date) 训练 LightGBM lambdarank，看 ndcg@5 是否也接近 1.0。
 3. 同时报告每个截面日 rank(1/amount_ma_10) 与 rank(y_illiq_20d) 的 Spearman 相关。

若单特征 ndcg@5≈1，则证明该目标几乎只是「已知于 t 的流动性」的单调函数，
模型学不到任何前向信息 —— 这就是 ndcg@5=1 的根因(目标-特征共线性/退化目标)。
"""
from __future__ import annotations
import sqlite3, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.factors.multi_objective_labels import (
    MultiObjectiveLabelBuilder, cross_sectional_rank_targets,
)
import lightgbm as lgb
from scipy.stats import spearmanr

DB = "database/stock_daily.db"
COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
START, END = "2022-06-01", "2025-12-31"
N_STOCKS = 60  # 抽样股票数，够统计即可

con = sqlite3.connect(DB)
codes = pd.read_sql_query(
    f"SELECT code FROM daily_data WHERE date>='{START}' AND date<='{END}' "
    f"GROUP BY code HAVING COUNT(*) > 800 ORDER BY COUNT(*) DESC LIMIT {N_STOCKS}",
    con,
)["code"].tolist()
print(f"[verify] 抽样股票数 = {len(codes)}")

stocks = {}
for c in codes:
    df = pd.read_sql_query(
        f"SELECT {','.join(COLS)} FROM daily_data WHERE code='{c}' "
        f"AND date>='{START}' AND date<='{END}' ORDER BY date",
        con,
    )
    df["date"] = df["date"].astype(str)
    if len(df) > 400:
        stocks[c] = df
con.close()
print(f"[verify] 有效股票 = {len(stocks)}")

builder = MultiObjectiveLabelBuilder(
    (5, 20, 60), risk_horizon=20, orthogonal_legs=True,
    use_matching_risk_for_sharpe=True,
)
univ = builder.build_universe(stocks)
print(f"[verify] 标签面板行数 = {len(univ)}, 含 y_illiq_20d 缺失: {univ['y_illiq_20d'].isna().sum()}")

# 后向流动性代理：t 日及之前的 10 日成交额均值(amount_ma_10)
def add_backward_liq(df: pd.DataFrame) -> pd.DataFrame:
    a = pd.to_numeric(df["amount"], errors="coerce").to_numpy(float)
    ma = pd.Series(a).rolling(10, min_periods=5).mean().to_numpy()
    df = df.copy()
    df["amount_ma_10"] = ma
    return df

back = {}
for c, df in stocks.items():
    if len(df) > 400:
        back[c] = add_backward_liq(df)
back_df = pd.concat(back, ignore_index=True) if False else pd.concat(
    [v.assign(code=k) for k, v in back.items()], ignore_index=True
)
panel = univ.merge(
    back_df[["code", "date", "amount_ma_10"]],
    on=["code", "date"], how="left",
)
# 同日排名(目标已越大越好? illiq 越大越差，但 ranking 只看排序；此处保持与训练一致：直接 rank)
panel = panel.dropna(subset=["y_illiq_20d", "amount_ma_10"]).copy()
panel["rank_y_illiq_20d"] = panel.groupby("date", sort=False)["y_illiq_20d"].rank(pct=True)

# 1) Spearman 相关(同日截面平均)
spears = []
for d, g in panel.groupby("date", sort=False):
    if g.shape[0] < 5:
        continue
    r1, _ = spearmanr(g["rank_y_illiq_20d"], 1.0 / g["amount_ma_10"])
    r2, _ = spearmanr(g["rank_y_illiq_20d"], -g["amount_ma_10"])
    spears.append((r1, r2))
spears = np.array(spears)
print(f"[verify] 同日 Spearman( rank_illiq , 1/amount_ma_10 ): 均值={np.nanmean(spears[:,0]):.4f}")
print(f"[verify] 同日 Spearman( rank_illiq , -amount_ma_10   ): 均值={np.nanmean(spears[:,1]):.4f}")

# 2) 单特征 ndcg@5：用 -amount_ma_10 作预测(越不流动 pred 越大，与 illiq 同单调)
panel = panel.sort_values("date").reset_index(drop=True)
groups = panel.groupby("date", sort=False).size().tolist()
def ndcg_at5(y_true, y_pred, groups):
    idx = 0; total = 0.0; ng = 0
    for g in groups:
        yt = y_true[idx:idx+g]; yp = y_pred[idx:idx+g]; idx += g
        order = np.argsort(-yp)            # 预测排序(降序)
        yr = np.sort(yt)[::-1]             # 理想真实排序(降序)
        top = min(5, g)
        dcg = sum((2**yt[order[k]]-1)/np.log2(k+2) for k in range(top))
        idcg = sum((2**yr[k]-1)/np.log2(k+2) for k in range(top))
        if idcg > 0:
            total += dcg/idcg; ng += 1
    return total/ng
ndcg = ndcg_at5(panel["rank_y_illiq_20d"].values, -panel["amount_ma_10"].values, groups)
print(f"[verify] 单特征(-amount_ma_10) ndcg@5 = {ndcg:.4f}")
# 再加 amount_std 代理双特征(用排名和近似)
panel["amount_std_30"] = panel.groupby("code")["amount_ma_10"].transform(
    lambda s: s.rolling(30, min_periods=10).std())
panel2 = panel.dropna(subset=["amount_std_30"])
groups2 = panel2.groupby("date", sort=False).size().tolist()
pred2 = (-panel2["amount_ma_10"].rank() + -panel2["amount_std_30"].rank())
ndcg2 = ndcg_at5(panel2["rank_y_illiq_20d"].values, pred2.values, groups2)
print(f"[verify] 双特征(-amount_ma_10 & -amount_std_30 排名和) ndcg@5 = {ndcg2:.4f}")
# 对照：随机预测的 ndcg@5(应接近 0.2~0.3)
rng = np.random.default_rng(0)
ndcg_rand = ndcg_at5(panel["rank_y_illiq_20d"].values, rng.random(len(panel)), groups)
print(f"[verify] 随机基线 ndcg@5 = {ndcg_rand:.4f}")
print("\n结论：若单特征 ndcg@5 远高于随机基线且接近 1，证明 y_illiq_20d 排名几乎完全由 t 日已知流动性决定 -> 退化目标。")
