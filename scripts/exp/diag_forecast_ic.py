"""T089 / E18 第 0 步：业绩预告特征的**零训练**信息量筛查。

动机（见 TRAINING_ITERATIONS.md E16 效率前沿 / T088）：打分层所有重加权轴已关闭，
唯一能外推前沿的是库外新信息。T088 已把业绩预告落库（106,589 行 / 5,453 只）。
在付出「4 次训练 + 多折 IC 判定」之前，先用**零训练**代价回答三个问题：

1. 覆盖率：任一交易日有多少比例的股票存在有效的、未过期的预告？
2. 单因子信息量：预告特征自身对 7 日前向收益的日度 Rank IC 是多少？
   并按 T082 铁律做**行情分层**（上涨日 / 下跌日），避免合并 IC 掩盖符号翻转。
3. 增量信息量：把特征对已在用的 `sue` / `YOYNI`（事后盈利同比）做截面回归后，
   残差还剩多少 IC —— 这决定它是「新信息」还是「已有列的换皮」。

PIT 纪律：只用 `pub_date`（披露日），绝不用 `stat_date`（报告期）。
特征在 d 日只能看到 `pub_date <= d` 的记录。标签为 `close.shift(-7)/close - 1`
（未复权，与训练管线同口径；截面 rank 后为共模偏差，见 T087）。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAILY_DB = os.path.join(ROOT, "database", "stock_daily.db")
FIN_DB = os.path.join(ROOT, "database", "stock_finance.db")
META_DB = os.path.join(ROOT, "database", "stock_meta.db")
CACHE_DIR = os.path.join(ROOT, "database", "system_data", "factors_cache")

FUTURE_DAYS = 7

# 预告类型 → 方向分数。正=盈利改善，负=恶化。数值只用于截面排序，绝对刻度无意义。
# 覆盖库内全部 11 个非空类型（T088 落库统计）：
#   预增 28418 / 略增 18837 / 预减 11911 / 首亏 11821 / 扭亏 9223 / 略减 8957 /
#   减亏 4963 / 增亏 4473 / 续盈 4295 / 续亏 1984 / 不确定 1451（另有 256 行空类型）
TYPE_SCORE: Dict[str, float] = {
    "预增": 2.0, "略增": 1.0, "扭亏": 1.5, "续盈": 0.5, "减亏": 0.5,
    "略减": -1.0, "预减": -2.0, "增亏": -1.5, "首亏": -2.0, "续亏": -2.0,
    "不确定": 0.0,
}

WINDOWS = {
    # 训练期验证区（三折验证窗的并集，样本内，用于筛选）
    "val": ("2017-10-18", "2022-09-05"),
    # 样本外，仅作证据，不参与筛选（拿 OOS 挑轴 = 把确认集烧成训练集）
    "oos_bear": ("2022-09-05", "2024-08-05"),
    "oos_bull": ("2024-08-05", "2026-08-05"),
}


def _universe(n: int, offset: int = 0) -> List[str]:
    with sqlite3.connect(META_DB) as c:
        codes = [r[0] for r in c.execute("SELECT code FROM stock_basic ORDER BY code")]
    return codes[offset:offset + n]


def load_prices(codes: List[str], start: str, end: str) -> pd.DataFrame:
    """读取日线，返回 long 表 code/date/close/tradestatus/is_st。"""
    out = []
    with sqlite3.connect(DAILY_DB) as c:
        for i in range(0, len(codes), 400):
            chunk = codes[i:i + 400]
            ph = ",".join("?" * len(chunk))
            q = (f"SELECT code,date,close,tradestatus,is_st FROM daily_data "
                 f"WHERE code IN ({ph}) AND date>=? AND date<=? ")
            out.append(pd.read_sql(q, c, params=[*chunk, start, end]))
    df = pd.concat(out, ignore_index=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def load_forecast() -> pd.DataFrame:
    with sqlite3.connect(FIN_DB) as c:
        df = pd.read_sql(
            "SELECT code,pub_date,stat_date,profitForcastType AS ftype,"
            "profitForcastChgPctUp AS up,profitForcastChgPctDwn AS dwn "
            "FROM performance_forecast WHERE pub_date IS NOT NULL AND pub_date!=''", c)
    df["up"] = pd.to_numeric(df["up"], errors="coerce")
    df["dwn"] = pd.to_numeric(df["dwn"], errors="coerce")
    df["ftype"] = df["ftype"].fillna("").str.strip()
    df["type_score"] = df["ftype"].map(TYPE_SCORE)
    # 幅度中值：上下限都缺则 NaN；只有一侧则用该侧
    df["chg_mid"] = df[["up", "dwn"]].mean(axis=1, skipna=True)
    df = df.sort_values(["code", "pub_date"]).reset_index(drop=True)
    return df


def build_features(prices: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """按 (code, date) 前向对齐最近一条预告（pub_date <= date），生成特征列。"""
    px = prices[["code", "date"]].copy()
    px["_d"] = pd.to_datetime(px["date"])
    f = fc.copy()
    f["_d"] = pd.to_datetime(f["pub_date"])
    # 显式复制一列披露日，merge_asof 的 on 列会被消耗掉，靠它算特征年龄
    f["pub_d"] = f["_d"]
    f = f[["code", "_d", "pub_d", "type_score", "chg_mid", "up", "dwn"]]

    merged = pd.merge_asof(
        px.sort_values("_d"), f.sort_values("_d"),
        on="_d", by="code", direction="backward", allow_exact_matches=True,
    )
    merged["fc_age_days"] = (merged["_d"] - merged["pub_d"]).dt.days

    out = merged[["code", "date", "type_score", "chg_mid", "fc_age_days"]].copy()
    # 事件窗口版本：只在预告发布后 60 个自然日内有效，之后视为过期（信息已被价格吸收）
    fresh = out["fc_age_days"] <= 60
    out["fc_type_60d"] = np.where(fresh, out["type_score"], np.nan)
    out["fc_chg_60d"] = np.where(fresh, out["chg_mid"], np.nan)
    # 未过期 + 幅度经 log1p 压缩（幅度分布极厚尾，截面 rank 前先压一层不改序但便于残差回归）
    out["fc_chg_log"] = np.sign(out["fc_chg_60d"]) * np.log1p(out["fc_chg_60d"].abs())
    out = out.rename(columns={"type_score": "fc_type_any", "chg_mid": "fc_chg_any"})
    out["fc_recency"] = np.where(out["fc_age_days"].notna(), -out["fc_age_days"], np.nan)
    return out


def add_labels(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["fwd"] = df.groupby("code")["close"].shift(-FUTURE_DAYS) / df["close"] - 1.0
    # 入场可买：非停牌
    df["buyable"] = df["tradestatus"].astype(str) == "1"
    return df


def load_control_cols(codes: List[str], cols: List[str]) -> pd.DataFrame:
    """从因子 parquet 缓存读取对照列（已在用的事后盈利同比等）。"""
    frames = []
    for code in codes:
        p = os.path.join(CACHE_DIR, f"{code}_factors.parquet")
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_parquet(p, columns=["date"] + cols)
        except Exception:
            continue
        d["code"] = code
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["code", "date"] + cols)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["date"].astype(str).str.slice(0, 10)
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    return (rankdata(x) - 0.5) / len(x)


def daily_ic(panel: pd.DataFrame, feat: str, min_valid: int = 50,
             residual_on: Optional[List[str]] = None) -> pd.DataFrame:
    """逐日 Spearman IC。只在特征非缺失的子集上算（覆盖子集口径）。

    residual_on 非空时，先在同一子集上把特征对对照列做 OLS（各列均截面 rank 化），
    用残差算 IC —— 回答「扣掉已有列之后还剩多少」。
    """
    rows = []
    need = ["code", "date", feat, "fwd"] + (residual_on or [])
    for date, g in panel[need].groupby("date", sort=True):
        g = g.dropna(subset=[feat, "fwd"])
        if residual_on:
            g = g.dropna(subset=residual_on)
        if len(g) < min_valid:
            continue
        x = _rank(g[feat].to_numpy(float))
        y = _rank(g["fwd"].to_numpy(float))
        if residual_on:
            Z = np.column_stack([_rank(g[c].to_numpy(float)) for c in residual_on])
            Z = np.column_stack([np.ones(len(Z)), Z])
            beta, *_ = np.linalg.lstsq(Z, x, rcond=None)
            x = x - Z @ beta
        if np.std(x) < 1e-12:
            continue
        ic = np.corrcoef(x, y)[0, 1]
        rows.append({"date": date, "ic": ic, "n": len(g)})
    return pd.DataFrame(rows)


def summarize(ic_df: pd.DataFrame, day_dir: pd.Series) -> Dict[str, object]:
    if ic_df.empty:
        return {"days": 0}
    d = ic_df.merge(day_dir.rename("dir"), left_on="date", right_index=True, how="left")
    res = {
        "days": int(len(d)),
        "mean_n": round(float(d["n"].mean()), 1),
        "ic_mean": round(float(d["ic"].mean()), 5),
        "ic_std": round(float(d["ic"].std()), 5),
        "ic_ir": round(float(d["ic"].mean() / d["ic"].std()), 3) if d["ic"].std() > 0 else None,
        "t_stat": round(float(d["ic"].mean() / (d["ic"].std() / np.sqrt(len(d)))), 2),
        "positive_ratio": round(float((d["ic"] > 0).mean()), 3),
    }
    for name, mask in (("up_days", d["dir"] > 0), ("down_days", d["dir"] <= 0)):
        sub = d[mask]
        res[name] = {
            "days": int(len(sub)),
            "ic_mean": round(float(sub["ic"].mean()), 5) if len(sub) else None,
            "t_stat": (round(float(sub["ic"].mean() / (sub["ic"].std() / np.sqrt(len(sub)))), 2)
                       if len(sub) > 5 and sub["ic"].std() > 0 else None),
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=800)
    ap.add_argument("--offset", type=int, default=0, help="股票池偏移，用于 holdout 池")
    ap.add_argument("--windows", default="val,oos_bear,oos_bull")
    ap.add_argument("--controls", default="sue,YOYNI,YOYPNI",
                    help="残差化对照列（来自因子 parquet 缓存）")
    ap.add_argument("--output", default=os.path.join(ROOT, "diagnose_output", "T089_forecast_ic.json"))
    args = ap.parse_args()

    codes = _universe(args.stocks, args.offset)
    wins = [w.strip() for w in args.windows.split(",") if w.strip()]
    start = min(WINDOWS[w][0] for w in wins)
    end = max(WINDOWS[w][1] for w in wins)
    # 标签要往后看 7 天，多取一段缓冲
    end_buf = (pd.Timestamp(end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"[1/5] 读取 {len(codes)} 只股票日线 {start}~{end_buf} ...", flush=True)
    px = load_prices(codes, start, end_buf)
    px = add_labels(px)
    print(f"      日线 {len(px):,} 行，{px['code'].nunique()} 只，"
          f"{px['date'].nunique()} 个交易日", flush=True)

    print("[2/5] 读取业绩预告并按 pub_date 前向对齐 ...", flush=True)
    fc = load_forecast()
    print(f"      预告 {len(fc):,} 行 / {fc['code'].nunique()} 只；"
          f"type_score 缺失 {int(fc['type_score'].isna().sum())}，"
          f"chg_mid 缺失 {int(fc['chg_mid'].isna().sum())}", flush=True)
    feats = build_features(px, fc)

    panel = px.merge(feats, on=["code", "date"], how="left")
    panel = panel[panel["buyable"]].copy()

    controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    if controls:
        print(f"[3/5] 读取对照列 {controls} ...", flush=True)
        ctrl = load_control_cols(codes, controls)
        panel = panel.merge(ctrl, on=["code", "date"], how="left")
        cov = {c: round(float(panel[c].notna().mean()), 4) for c in controls}
        print(f"      对照列覆盖率 {cov}", flush=True)

    feat_cols = ["fc_type_any", "fc_chg_any", "fc_type_60d", "fc_chg_60d",
                 "fc_chg_log", "fc_recency"]
    result: Dict[str, object] = {"meta": {
        "stocks": len(codes), "offset": args.offset, "future_days": FUTURE_DAYS,
        "controls": controls, "type_score_map": TYPE_SCORE,
    }, "windows": {}}

    print("[4/5] 逐窗口计算日度 IC ...", flush=True)
    for w in wins:
        w_start, w_end = WINDOWS[w]
        sub = panel[(panel["date"] >= w_start) & (panel["date"] <= w_end)].copy()
        sub = sub.dropna(subset=["fwd"])
        # 行情方向：当日全截面平均前向收益的符号（T082 口径）
        day_dir = sub.groupby("date")["fwd"].mean()
        n_days = sub["date"].nunique()
        wres: Dict[str, object] = {
            "range": [w_start, w_end], "days": int(n_days),
            "coverage": {}, "features": {},
        }
        for f in feat_cols:
            wres["coverage"][f] = round(float(sub[f].notna().mean()), 4)
        for f in feat_cols:
            ic = daily_ic(sub, f)
            entry = {"raw": summarize(ic, day_dir)}
            if controls and f in ("fc_type_any", "fc_chg_any", "fc_type_60d", "fc_chg_log"):
                icr = daily_ic(sub, f, residual_on=controls)
                entry["residual_vs_controls"] = summarize(icr, day_dir)
            wres["features"][f] = entry
            r = entry["raw"]
            print(f"  [{w}] {f:<12} cov={wres['coverage'][f]:.3f} "
                  f"days={r.get('days')} IC={r.get('ic_mean')} t={r.get('t_stat')} "
                  f"up={r.get('up_days', {}).get('ic_mean')} "
                  f"down={r.get('down_days', {}).get('ic_mean')}", flush=True)
        result["windows"][w] = wres

    print("[5/5] 写出 ...", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"完成 → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
