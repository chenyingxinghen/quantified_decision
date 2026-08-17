"""T089 / E18 第 1 步：把业绩预告信息**混进已冻结基线打分**，做零训练的边际价值判定。

背景：`scripts/exp/exp_nam_gate.py` 及其判定器在某次仓库清理中丢失（从未进 git），
所以「4 次训练 + 多折 IC」这条既有协议暂时跑不动。但要回答的问题——
「这列新信息能不能让排序更准」——不需要重训：把已存基线模型的当日打分分位与
预告特征分位做线性混合，看日度 Rank IC 怎么变。这与 T082/T084/T085 那三轮
「用已存模型做零训练诊断」是同一套手法，成本 0 训练 0 回测。

口径（与 `core/backtest/strategies/ml_factor_strategy.py` 的推理路径逐步对齐）：
1. 从因子 parquet 取模型自带的 `feature_names` 列；
2. 当日截面对非 skip-rank 列做 `rankdata/(n+1)`；
3. skip-rank 连续列用 `norm_stats['skip_col_stats']` 的 robust-sigmoid（T043 铁律）；
4. `nan_to_num(nan=0.5)` 后送入 `NAMGateModel.predict`；
5. 打分转当日分位 `score_pct`，与预告特征分位 `fc_pct` 按 `w` 线性混合。

判定按 [[regime-stratified-ic-gate]]：合并 ΔIC 之外**必须**分别看上涨日 / 下跌日，
下跌日一边倒为负即判 regime 倾斜、直接关轴。多模型（4 个基线种子）做配对。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from config import factor_config as fc  # noqa: E402
from core.factors.nam_gate_model import NAMGateModel  # noqa: E402
from scripts.exp.diag_forecast_ic import (  # noqa: E402
    FUTURE_DAYS, WINDOWS, add_labels, build_features, load_forecast, load_prices, _universe,
)

CACHE_DIR = os.path.join(ROOT, "database", "system_data", "factors_cache")
MODEL_ROOT = os.path.join(ROOT, "models", "nam_gate")

# 台账里的 4 个基线种子模型（T045 = seed42，T068_seed* 为同配置另 3 个种子）
BASE_MODELS = {
    42: "T045",
    11: "T068_seed11",
    23: "T068_seed23",
    37: "T068_seed37",
}


def _load_model(name: str) -> Tuple[NAMGateModel, dict]:
    d = os.path.join(MODEL_ROOT, name)
    pkl = os.path.join(d, "nam_gate_factor_model.pkl")
    model = NAMGateModel().load_model(pkl)
    ns_path = os.path.join(d, "norm_stats.pkl")
    if not os.path.exists(ns_path):
        raise FileNotFoundError(f"{name} 缺少 norm_stats.pkl —— T043 铁律，拒绝运行")
    import pickle
    with open(ns_path, "rb") as fh:
        norm_stats = pickle.load(fh)
    return model, norm_stats


def load_panel(codes: List[str], feature_names: List[str], start: str, end: str) -> pd.DataFrame:
    """读取因子面板（只取模型需要的列 + 窗口内日期）。"""
    frames = []
    missing_cols = 0
    for i, code in enumerate(codes):
        p = os.path.join(CACHE_DIR, f"{code}_factors.parquet")
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_parquet(p, columns=["date"] + feature_names)
        except Exception:
            missing_cols += 1
            continue
        d["date"] = d["date"].astype(str).str.slice(0, 10)
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        if d.empty:
            continue
        d["code"] = code
        frames.append(d)
        if (i + 1) % 200 == 0:
            print(f"      ... {i + 1}/{len(codes)}", flush=True)
    if missing_cols:
        print(f"      跳过 {missing_cols} 只（parquet 缺列）", flush=True)
    panel = pd.concat(frames, ignore_index=True)
    for c in feature_names:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(np.float32)
    return panel


def _robust_plan(norm_stats: dict, feature_names: List[str]):
    """预计算 skip-rank 连续列的 robust-sigmoid 方案，避免逐日重算索引。"""
    stats = (norm_stats or {}).get("skip_col_stats")
    if not stats or stats.get("robust_global_idx", np.array([])).size == 0:
        return []
    train_names = norm_stats.get("factor_names", [])
    pos = {n: i for i, n in enumerate(feature_names)}
    plan = []
    for j, gi in enumerate(stats["robust_global_idx"]):
        if gi >= len(train_names):
            continue
        col = train_names[gi]
        if col not in pos:
            continue
        plan.append((pos[col], float(stats["median"][j]), float(stats["iqr"][j]),
                     bool(stats["valid_iqr"][j])))
    return plan


def score_panel(panel: pd.DataFrame, feature_names: List[str], model: NAMGateModel,
                norm_stats: dict) -> pd.Series:
    """逐日截面归一化 + 打分，返回与 panel 同序的当日打分分位。"""
    rank_idx = [i for i, c in enumerate(feature_names) if not fc.TrainingConfig.should_skip_rank(c)]
    plan = _robust_plan(norm_stats, feature_names)
    X_all = panel[feature_names].to_numpy(dtype=np.float32, copy=False)
    out = np.full(len(panel), np.nan, dtype=np.float64)
    day_pos = panel.groupby("date", sort=True).indices
    for k, (date, idx) in enumerate(day_pos.items()):
        if len(idx) < 30:
            continue
        X = X_all[idx].copy()
        if rank_idx:
            X[:, rank_idx] = (rankdata(X[:, rank_idx], method="average", axis=0)
                              / (len(X) + 1)).astype(np.float32)
        for col_idx, med, iqr, valid in plan:
            if valid:
                z = (X[:, col_idx].astype(np.float64) - med) / iqr
                X[:, col_idx] = (1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))).astype(np.float32)
            else:
                X[:, col_idx] = 0.0
        X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)
        model.set_context_date(date)
        pred = np.asarray(model.predict(pd.DataFrame(X, columns=feature_names)), dtype=float)
        out[idx] = rankdata(pred, method="average") / (len(pred) + 1)
        if (k + 1) % 250 == 0:
            print(f"      打分 {k + 1}/{len(day_pos)} 日", flush=True)
    return pd.Series(out, index=panel.index, name="score_pct")


def _pct(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average") / (len(x) + 1)


def blend_ic(df: pd.DataFrame, feat: str, weights: List[float],
             topk: int = 40, min_valid: int = 50) -> Dict[str, Dict[str, object]]:
    """逐日算 base / blend 的 Rank IC 与 top-K 超额，按行情方向分层汇总。"""
    recs = []
    for date, g in df.groupby("date", sort=True):
        g = g.dropna(subset=["score_pct", "fwd"])
        if len(g) < min_valid:
            continue
        y = g["fwd"].to_numpy(float)
        y_pct = _pct(y)
        s = g["score_pct"].to_numpy(float)
        has_f = g[feat].notna().to_numpy()
        f_raw = g[feat].to_numpy(float)
        # 缺失置中位（信息不足时中性，不给假极值），与 E7 的 0.5 置中同思路
        f_pct = np.full(len(g), 0.5)
        if has_f.sum() >= 10:
            f_pct[has_f] = _pct(f_raw[has_f])
        row = {"date": date, "n": len(g), "mean_fwd": float(y.mean()),
               "cov": float(has_f.mean())}
        for w in weights:
            blended = (1.0 - w) * s + w * f_pct
            row[f"ic_{w}"] = float(np.corrcoef(_pct(blended), y_pct)[0, 1])
            k = min(topk, max(1, len(g) // 10))
            top = np.argsort(-blended)[:k]
            row[f"tk_{w}"] = float(y[top].mean() - y.mean())
        recs.append(row)
    d = pd.DataFrame(recs)
    res: Dict[str, Dict[str, object]] = {}
    up, dn = d["mean_fwd"] > 0, d["mean_fwd"] <= 0
    for w in weights:
        entry = {"days": int(len(d)), "coverage": round(float(d["cov"].mean()), 4)}
        for seg, mask in (("all", np.ones(len(d), bool)), ("up_days", up.to_numpy()),
                          ("down_days", dn.to_numpy())):
            sub = d[mask]
            ic, tk = sub[f"ic_{w}"], sub[f"tk_{w}"]
            base_ic = sub[f"ic_{weights[0]}"]
            dif = ic - base_ic
            entry[seg] = {
                "days": int(len(sub)),
                "rank_ic": round(float(ic.mean()), 5),
                "d_rank_ic": round(float(dif.mean()), 5),
                # 配对（同日同模型）t 值：日间自相关未修正，7 日重叠标签下偏乐观，只作方向指示
                "d_t_paired": (round(float(dif.mean() / (dif.std() / np.sqrt(len(dif)))), 2)
                               if len(dif) > 5 and dif.std() > 0 else None),
                "d_positive_ratio": round(float((dif > 0).mean()), 3),
                f"top{topk}_excess": round(float(tk.mean()), 6),
                f"d_top{topk}_excess": round(float((tk - sub[f'tk_{weights[0]}']).mean()), 6),
            }
        res[str(w)] = entry
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=800)
    ap.add_argument("--seeds", default="42,11,23,37")
    ap.add_argument("--feature", default="fc_type_60d",
                   help="混合用的预告特征，逗号分隔可一次跑多列（见 diag_forecast_ic.py）")
    ap.add_argument("--weights", default="0,0.05,0.1,0.2")
    ap.add_argument("--windows", default="val,oos_bear,oos_bull")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--output", default=os.path.join(ROOT, "diagnose_output", "T089_score_blend.json"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    features = [f.strip() for f in args.feature.split(",") if f.strip()]
    weights = [float(w) for w in args.weights.split(",")]
    if weights[0] != 0.0:
        raise SystemExit("weights 首项必须是 0（作为同日配对的基线臂）")
    wins = [w.strip() for w in args.windows.split(",")]
    start = min(WINDOWS[w][0] for w in wins)
    end = max(WINDOWS[w][1] for w in wins)
    end_buf = (pd.Timestamp(end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    codes = _universe(args.stocks)
    print(f"[1/4] 标签与预告特征 ...", flush=True)
    px = add_labels(load_prices(codes, start, end_buf))
    feats = build_features(px, load_forecast())
    lab = px.merge(feats, on=["code", "date"], how="left")
    lab = lab[lab["buyable"]][["code", "date", "fwd"] + features]

    print(f"[2/4] 载入模型 {seeds} 并读取因子面板 ...", flush=True)
    models = {}
    for s in seeds:
        m, ns = _load_model(BASE_MODELS[s])
        models[s] = (m, ns)
    feature_names = list(models[seeds[0]][0].feature_names)
    print(f"      特征 {len(feature_names)} 列", flush=True)
    panel = load_panel(codes, feature_names, start, end_buf)
    print(f"      面板 {len(panel):,} 行 / {panel['code'].nunique()} 只 / "
          f"{panel['date'].nunique()} 日", flush=True)

    result = {"meta": {"stocks": args.stocks, "seeds": seeds, "features": features,
                       "weights": weights, "topk": args.topk,
                       "future_days": FUTURE_DAYS, "base_models": BASE_MODELS},
              "seeds": {}}

    for s in seeds:
        model, ns = models[s]
        print(f"[3/4] seed {s} ({BASE_MODELS[s]}) 打分 ...", flush=True)
        panel["score_pct"] = score_panel(panel, feature_names, model, ns).to_numpy()
        merged = panel[["code", "date", "score_pct"]].merge(lab, on=["code", "date"], how="inner")
        merged = merged.dropna(subset=["fwd"])
        res_s = {}
        for w in wins:
            a, b = WINDOWS[w]
            sub = merged[(merged["date"] >= a) & (merged["date"] <= b)]
            res_s[w] = {}
            for feat in features:
                res_s[w][feat] = blend_ic(sub, feat, weights, topk=args.topk)
                for wt in weights:
                    e = res_s[w][feat][str(wt)]
                    print(f"  [s{s}][{w}][{feat}] w={wt}: IC={e['all']['rank_ic']} "
                          f"(Δ{e['all']['d_rank_ic']}, t={e['all']['d_t_paired']}) "
                          f"up Δ{e['up_days']['d_rank_ic']} down Δ{e['down_days']['d_rank_ic']} "
                          f"top{args.topk}Δ={e['all'][f'd_top{args.topk}_excess']}", flush=True)
        result["seeds"][str(s)] = res_s
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)

    print("[4/4] 汇总（跨种子中位 + 符号计数）", flush=True)
    summary: Dict[str, dict] = {}
    for w in wins:
        summary[w] = {}
        for feat in features:
            summary[w][feat] = {}
            for wt in weights[1:]:
                for seg in ("all", "up_days", "down_days"):
                    vals = [result["seeds"][str(s)][w][feat][str(wt)][seg]["d_rank_ic"]
                            for s in seeds]
                    summary[w][feat].setdefault(str(wt), {})[seg] = {
                        "d_ic_median": round(float(np.median(vals)), 5),
                        "positive_seeds": int(sum(v > 0 for v in vals)),
                        "per_seed": vals,
                    }
    result["summary"] = summary
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"完成 → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
