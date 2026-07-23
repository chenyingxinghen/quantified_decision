"""验证：y_illiq_20d 改为「流动性变化」目标(illiq_future - illiq_now)后是否仍退化。

对比旧水平目标的退化指标：
  - 单特征 -amount_ma_10 的 ndcg@5（旧=0.9055，随机=0.4679）
  - 同日截面 Spearman(rank_illiq, 1/amount_ma_10)（旧=0.806）

若新目标的单特征 ndcg@5 远低于 0.9、Spearman 远低于 0.8，则退化性已消失，
只剩流动性变化的真实前向信息（可能被模型学到，但不会是确定性的）。
"""
from __future__ import annotations
import sqlite3, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.factors.multi_objective_labels import (
    MultiObjectiveLabelBuilder, cross_sectional_rank_targets,
)
from scipy.stats import spearmanr

DB = "database/stock_daily.db"
COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
START, END = "2022-06-01", "2025-12-31"
N_STOCKS = 60

con = sqlite3.connect(DB)
codes = pd.read_sql_query(
    f"SELECT code FROM daily_data WHERE date>='{START}' AND date<='{END}' "
    f"GROUP BY code HAVING COUNT(*) > 800 ORDER BY COUNT(*) DESC LIMIT {N_STOCKS}",
    con,
)["code"].tolist()
stocks = {}
for c in codes:
    df = pd.read_sql_query(
        f"SELECT {','.join(COLS)} FROM daily_data WHERE code='{c}' "
        f"AND date>='{START}' AND date<='{END}' ORDER BY date", con,
    )
    df["date"] = df["date"].astype(str)
    if len(df) > 400:
        stocks[c] = df
con.close()

builder = MultiObjectiveLabelBuilder(
    (5, 20, 60), risk_horizon=20, orthogonal_legs=True,
    use_matching_risk_for_sharpe=True,
)
univ = builder.build_universe(stocks)
print(f"[chg] 面板行数={len(univ)}  新 y_illiq_20d 缺失={univ['y_illiq_20d'].isna().sum()}")
v = univ["y_illiq_20d"].to_numpy(float)
print(f"[chg] y_illiq_20d 统计: mean={np.nanmean(v):.3e} std={np.nanstd(v):.3e} "
      f"min={np.nanmin(v):.3e} max={np.nanmax(v):.3e}")

# 后向流动性代理
def add_back(df):
    a = pd.to_numeric(df["amount"], errors="coerce").to_numpy(float)
    df = df.copy()
    df["amount_ma_10"] = pd.Series(a).rolling(10, min_periods=5).mean().to_numpy()
    return df
back = {k: add_back(v) for k, v in stocks.items()}
back_df = pd.concat([x.assign(code=k) for k, x in back.items()], ignore_index=True)
panel = univ.merge(back_df[["code", "date", "amount_ma_10"]], on=["code", "date"], how="left")
panel = panel.dropna(subset=["y_illiq_20d", "amount_ma_10"]).copy()
# 训练方向：illiq 为风险列 -> ascending=False（变化越大=越不流动=越低分）
panel["rank_y_illiq_20d"] = panel.groupby("date", sort=False)["y_illiq_20d"].rank(
    pct=True, ascending=False)

# 1) 同日 Spearman（变化目标 vs 后向流动性水平）
sp = []
for d, g in panel.groupby("date", sort=False):
    if g.shape[0] < 5:
        continue
    r1, _ = spearmanr(g["rank_y_illiq_20d"], 1.0 / g["amount_ma_10"])
    r2, _ = spearmanr(g["y_illiq_20d"], g["amount_ma_10"])
    sp.append((r1, r2))
sp = np.array(sp)
print(f"[chg] 同日 Spearman(rank_illiq_chg, 1/amount_ma_10): 均值={np.nanmean(sp[:,0]):.4f}")
print(f"[chg] 同日 Spearman(y_illiq_chg, amount_ma_10)       : 均值={np.nanmean(sp[:,1]):.4f}")

# 2) 单特征 ndcg@5（预测 = -amount_ma_10，越不流动 pred 越大）
panel = panel.sort_values("date").reset_index(drop=True)
groups = panel.groupby("date", sort=False).size().tolist()
def ndcg_at5(y_true, y_pred, groups):
    idx = 0; total = 0.0; ng = 0
    for g in groups:
        yt = y_true[idx:idx+g]; yp = y_pred[idx:idx+g]; idx += g
        order = np.argsort(-yp)
        yr = np.sort(yt)[::-1]
        top = min(5, g)
        dcg = sum((2**yt[order[k]]-1)/np.log2(k+2) for k in range(top))
        idcg = sum((2**yr[k]-1)/np.log2(k+2) for k in range(top))
        if idcg > 0:
            total += dcg/idcg; ng += 1
    return total/ng
ndcg = ndcg_at5(panel["rank_y_illiq_20d"].values, -panel["amount_ma_10"].values, groups)
rng = np.random.default_rng(0)
ndcg_rand = ndcg_at5(panel["rank_y_illiq_20d"].values, rng.random(len(panel)), groups)
print(f"[chg] 单特征(-amount_ma_10) ndcg@5 = {ndcg:.4f}   (旧水平目标=0.9055, 随机={ndcg_rand:.4f})")

print("\n结论：若新目标单特征 ndcg@5 远低于 0.9（且显著高于随机），说明退化性已消除，"
      "目标只剩流动性变化的真实前向信息。")
