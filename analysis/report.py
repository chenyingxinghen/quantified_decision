"""
analysis/report.py — 报告与可视化装配

生成:
  - <out>/report.md            结构化中文报告（含各指标结论与图表引用）
  - <out>/metrics.csv          关键指标扁平表
  - <out>/figures/*.png        所有可视化图表（matplotlib Agg 后端）

字体策略:
  - 优先注册系统中文字体（SimHei / Microsoft YaHei / Noto Sans CJK 等），
    图表使用中文标签；
  - 若云端/离线环境无中文字体，自动回退为英文标签，避免方框乱码，保证图表可渲染。

报告结构严格对应需求清单:
  稳健性检验与拓展分析（统计性/经济性/投资组合/交易成本）
  SHAP 方法框架与全局特征重要性
  财务/交易/估值 三类特征边际贡献
  机制分析（交互/一致性/A股定价规律）
"""
from __future__ import annotations

import os
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.autolayout": True})


# ── 字体引导：优先 CJK，否则英文回退 ─────────────────────────────────────
def _setup_cjk_font() -> bool:
    candidates = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wenquanyi/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/droidsansfallback/DroidSansFallback.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
                prop = fm.FontProperties(fname=fp)
                plt.rcParams["font.family"] = prop.get_name()
                plt.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                continue
    return False


CJK_OK = _setup_cjk_font()

# 标签字典：有中文字体用 zh，否则用英文（保证图表在无 CJK 环境也能渲染）
L = {
    "rankic_ts_title": "RankIC 时间序列" if CJK_OK else "RankIC Time Series",
    "rankic_daily": "每日 RankIC" if CJK_OK else "Daily RankIC",
    "rankic_ma": "60日均值" if CJK_OK else "60d Mean",
    "ic_regime_title": "各市场状态 RankIC 均值" if CJK_OK else "RankIC by Market Regime",
    "bull": "牛市" if CJK_OK else "Bull",
    "bear": "熊市" if CJK_OK else "Bear",
    "sideways": "震荡" if CJK_OK else "Sideways",
    "grouping_title": "十分位分组平均收益" if CJK_OK else "Decile Group Mean Return",
    "grouping_x": ("收益分组（0=低分尾组, 9=高分头组）"
                   if CJK_OK else "Return Group (0=loser, 9=winner)"),
    "grouping_y": "平均日收益 (%)" if CJK_OK else "Mean Daily Return (%)",
    "ls_title": "多空组合累计净值" if CJK_OK else "Long-Short Cumulative Net Value",
    "ls_gross": "多空组合（未扣费）" if CJK_OK else "Long-Short (gross)",
    "ls_net": "多空组合（扣费后）" if CJK_OK else "Long-Short (net)",
    "turnover_title": "组合换手率（多空双边）" if CJK_OK else "Portfolio Turnover (both sides)",
    "turnover_y": "双边换手率 (%)" if CJK_OK else "Two-sided Turnover (%)",
    "turnover_mean": "均值 {x}%" if CJK_OK else "Mean {x}%",
    "shap_title": "Top {n} 全局特征重要性（SHAP）" if CJK_OK else "Top {n} Global Feature Importance (SHAP)",
    "cat_title": "三类特征边际贡献" if CJK_OK else "Marginal Contribution by Category",
    "cat_y": "边际贡献占比 (%)" if CJK_OK else "Marginal Contribution (%)",
    "cat_fin": "财务" if CJK_OK else "Financial",
    "cat_trd": "交易" if CJK_OK else "Trading",
    "cat_val": "估值" if CJK_OK else "Valuation",
    "cat_oth": "其他" if CJK_OK else "Other",
    "interaction_title": "Top 特征 SHAP 交互强度" if CJK_OK else "Top-feature SHAP Interaction",
    "consistency_title": ("模型间 SHAP 一致性（特征重要性排序 Spearman）"
                          if CJK_OK else "Cross-model SHAP Consistency (Spearman)"),
}


# ── 配色（红涨绿跌，符合 A 股习惯）────────────────────────────────────────
C_UP = "#d62728"
C_DOWN = "#2ca02c"
C_LINE = "#1f77b4"
C_ACC = "#ff7f0e"


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# 图表
# ════════════════════════════════════════════════════════════════════════
def fig_rankic_ts(daily_ic: pd.Series, path: str):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(daily_ic.index, daily_ic.values, lw=0.6, color=C_LINE, label=L["rankic_daily"])
    roll = daily_ic.rolling(60, min_periods=10).mean()
    ax.plot(roll.index, roll.values, lw=1.6, color=C_ACC, label=L["rankic_ma"])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(L["rankic_ts_title"])
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, path)


def fig_ic_by_regime(regime: dict, path: str):
    by = regime.get("by_regime", {})
    labels, vals = [], []
    for k in ["bull", "bear", "sideways"]:
        d = by.get(k, {})
        if "rankic_mean" in d:
            labels.append(L[k])
            vals.append(d["rankic_mean"])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(labels, vals, color=[C_UP, C_DOWN, "#7f7f7f"])
    ax.set_title(L["ic_regime_title"])
    ax.axhline(0, color="grey", lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    _save(fig, path)


def fig_grouping_bar(group_mean_ret: pd.Series, path: str):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = group_mean_ret.index.astype(int)
    colors = [C_DOWN if i < len(x) / 2 else C_UP for i in x]
    ax.bar(x, group_mean_ret.values * 100, color=colors)
    ax.set_xlabel(L["grouping_x"])
    ax.set_ylabel(L["grouping_y"])
    ax.set_title(L["grouping_title"])
    _save(fig, path)


def fig_ls_cum(ls_cum: pd.Series, net_cum: pd.Series, path: str):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    if ls_cum is not None and len(ls_cum):
        ax.plot(ls_cum.index, ls_cum.values, color=C_UP, lw=1.2, label=L["ls_gross"])
    if net_cum is not None and len(net_cum):
        ax.plot(net_cum.index, net_cum.values, color=C_DOWN, lw=1.2, label=L["ls_net"])
    ax.set_title(L["ls_title"])
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)


def fig_turnover(ts: pd.Series, path: str):
    if ts is None or len(ts) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(ts.index, ts.values * 100, color=C_LINE, lw=0.7)
    ax.axhline(ts.mean() * 100, color=C_ACC, lw=1.2, ls="--",
               label=L["turnover_mean"].format(x=f"{ts.mean()*100:.1f}"))
    ax.set_ylabel(L["turnover_y"])
    ax.set_title(L["turnover_title"])
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_shap_top(importance: dict, path: str, top: int = 20):
    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top]
    if not items:
        return
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, vals, color=C_LINE)
    ax.set_title(L["shap_title"].format(n=top))
    _save(fig, path)


def fig_category_share(share: dict, path: str):
    labels = [L["cat_fin"], L["cat_trd"], L["cat_val"], L["cat_oth"]]
    vals = [share.get("财务", 0.0) * 100, share.get("交易", 0.0) * 100,
            share.get("估值", 0.0) * 100, share.get("其他", 0.0) * 100]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(labels, vals, color=[C_LINE, C_ACC, C_UP, "#7f7f7f"])
    ax.set_ylabel(L["cat_y"])
    ax.set_title(L["cat_title"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    _save(fig, path)


def fig_interaction(matrix, features, path: str):
    if matrix is None or len(features) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Reds")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_yticklabels(features, fontsize=7)
    ax.set_title(L["interaction_title"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def fig_consistency(pairs: dict, path: str):
    if not pairs:
        return
    keys = list(pairs.keys())
    vals = [pairs[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(keys, vals, color=C_LINE)
    ax.set_xlim(0, 1)
    ax.set_title(L["consistency_title"])
    _save(fig, path)


# ════════════════════════════════════════════════════════════════════════
# 指标表 & 报告
# ════════════════════════════════════════════════════════════════════════
def build_metrics_csv(results: dict, path: str) -> dict:
    robust = results["robustness"]
    port = results["portfolio"]
    shap_r = results["shap"]
    if shap_r is None or shap_r.get("skipped"):
        shap_r = {
            "skipped": True,
            "global": {"method": "neural-skip" if (shap_r or {}).get("reason") == "neural" else "disabled"},
            "category": {"share": {}},
            "consistency": {"mean_consistency": float("nan")},
        }
    st = robust["stats"]
    regime = robust["regime"]["by_regime"]
    cat = shap_r["category"]["share"]
    tcost = port["tcost"]
    cons = shap_r["consistency"]

    row = {
        "rankic_mean": round(st["rankic_mean"], 4),
        "rankic_std": round(st["rankic_std"], 4),
        "icir": round(st["icir"], 4),
        "ic_t_stat": round(st["t_stat"], 4),
        "ic_p_value": round(st["p_value"], 6),
        "ic_positive_ratio": round(st["ic_positive_ratio"], 4),
        "ic_significant_5pct": st["significant_5pct"],
        "regime_bull_ic": round(regime.get("bull", {}).get("rankic_mean", float("nan")), 4),
        "regime_bear_ic": round(regime.get("bear", {}).get("rankic_mean", float("nan")), 4),
        "regime_sideways_ic": round(regime.get("sideways", {}).get("rankic_mean", float("nan")), 4),
        "f_test_p_value": round(robust["regime"].get("f_test", {}).get("p_value", float("nan")), 6),
        "ls_ann_return_gross": round(tcost["gross"].get("ann_return", float("nan")), 4),
        "ls_sharpe_gross": round(tcost["gross"].get("sharpe", float("nan")), 4),
        "ls_max_dd_gross": round(tcost["gross"].get("max_drawdown", float("nan")), 4),
        "ls_ann_return_net": round(tcost["net"].get("ann_return", float("nan")), 4),
        "ls_sharpe_net": round(tcost["net"].get("sharpe", float("nan")), 4),
        "cost_drag_ann": round(tcost["cost_drag_ann"], 4),
        "mean_turnover_total": round(port["turnover"]["mean_turnover_total"], 4),
        "cat_financial": round(cat.get("财务", 0.0), 4),
        "cat_trading": round(cat.get("交易", 0.0), 4),
        "cat_valuation": round(cat.get("估值", 0.0), 4),
        "shap_method": shap_r["global"]["method"],
        "cross_model_consistency": round(cons.get("mean_consistency", float("nan")), 4),
        "n_samples": results.get("meta", {}).get("n_samples", ""),
        "train_range": results.get("meta", {}).get("train_range", ""),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in row.items():
            w.writerow([k, v])
    return row


def write_report(results: dict, out_dir: str) -> dict:
    fig_dir = os.path.join(out_dir, "figures")
    robust = results["robustness"]
    port = results["portfolio"]
    shap_r = results["shap"]
    shap_skipped = bool(shap_r is None or shap_r.get("skipped"))
    if shap_skipped:
        # 神经网络模型（无树结构）或 --no-shap：SHAP 不适用，替换为占位结构，
        # 下方图表与章节生成会据此跳过 SHAP 专属内容。
        reason = (shap_r or {}).get("reason", "disabled")
        shap_r = {
            "skipped": True,
            "reason": reason,
            "global": {"method": "neural-skip" if reason == "neural" else "disabled",
                       "importance": {}},
            "category": {"method": "skipped", "share": {}, "shares": {}},
            "interaction": {"skipped": True},
            "consistency": {"skipped": True, "per_objective_rank_corr": {},
                            "mean_consistency": float("nan")},
        }
    mech = results["mechanism"]
    st = robust["stats"]
    regime = robust["regime"]
    tcost = port["tcost"]
    cat = shap_r["category"]

    # ── 生成图表 ──
    paths = {}
    paths["rankic_ts"] = os.path.join(fig_dir, "rankic_ts.png")
    fig_rankic_ts(robust["daily_ic"], paths["rankic_ts"])

    paths["ic_regime"] = os.path.join(fig_dir, "ic_by_regime.png")
    fig_ic_by_regime(regime, paths["ic_regime"])

    paths["grouping"] = os.path.join(fig_dir, "grouping_bar.png")
    fig_grouping_bar(port["grouping"]["group_mean_ret"], paths["grouping"])

    paths["ls_cum"] = os.path.join(fig_dir, "ls_cum.png")
    fig_ls_cum(port["grouping"]["ls_cum"], tcost["net_cum"], paths["ls_cum"])

    paths["turnover"] = os.path.join(fig_dir, "turnover.png")
    fig_turnover(port["turnover"]["turnover_total"], paths["turnover"])

    paths["shap_top"] = os.path.join(fig_dir, "shap_top.png")
    paths["category"] = os.path.join(fig_dir, "category_share.png")
    paths["interaction"] = os.path.join(fig_dir, "interaction.png")
    paths["consistency"] = os.path.join(fig_dir, "consistency.png")
    if not shap_skipped:
        fig_shap_top(shap_r["global"]["importance"], paths["shap_top"], top=20)
        fig_category_share(cat["share"], paths["category"])
        fig_interaction(shap_r["interaction"].get("matrix"),
                        shap_r["interaction"].get("features"), paths["interaction"])
        fig_consistency(shap_r["consistency"].get("per_objective_rank_corr", {}),
                        paths["consistency"])

    # ── 指标表 ──
    metrics_path = os.path.join(out_dir, "metrics.csv")
    row = build_metrics_csv(results, metrics_path)

    def fnum(x, p=4):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "N/A"

    # ── 报告正文 ──
    md = []
    md.append("# 量化因子模型 · 稳健性检验与拓展分析报告\n")
    meta = results.get("meta", {})
    md.append(
        f"> 样本量: **{meta.get('n_samples','?')}** | "
        f"训练区间: **{meta.get('train_range','?')}** | "
        f"SHAP 方法: **{shap_r['global']['method']}**\n"
    )

    md.append("## 一、稳健性检验与拓展分析\n")

    md.append("### 1.1 统计性检验（RankIC / ICIR / t / F 检验）\n")
    ic_sign = ("正向，模型得分与未来收益单调正相关"
               if st["rankic_mean"] > 0
               else "负值，模型得分与未来收益单调负相关（建议核对标签方向）")
    md.append(
        f"- **RankIC 均值**: {fnum(st['rankic_mean'])}（{ic_sign}）\n"
        f"- **RankIC 标准差**: {fnum(st['rankic_std'])}\n"
        f"- **ICIR（信息比率）**: {fnum(st['icir'])} — 越高说明 IC 稳定可持续\n"
        f"- **t 检验**: t = {fnum(st['t_stat'])}, p = {fnum(st['p_value'],6)} → "
        f"{'在 5% 水平显著异于 0' if st['significant_5pct'] else '未达显著'}\n"
        f"- **IC>0 占比**: {fnum(st['ic_positive_ratio']*100,2)}%（{st['n_days']} 个交易日）\n"
    )
    md.append(f"![RankIC 时间序列](figures/{os.path.basename(paths['rankic_ts'])})\n")

    md.append("### 1.2 经济性检验（牛熊震荡异质性 / SHAP 特征重要性）\n")
    rb = regime.get("by_regime", {})
    for k, label in [("bull", "牛市"), ("bear", "熊市"), ("sideways", "震荡")]:
        d = rb.get(k, {})
        if "rankic_mean" in d:
            md.append(
                f"- **{label}**: RankIC={fnum(d['rankic_mean'])}，ICIR={fnum(d.get('icir',float('nan')))}, "
                f"头尾组收益差={fnum(d.get('top_minus_bottom',float('nan'))*100,3)}%\n"
            )
        else:
            md.append(f"- **{label}**: 样本不足（{d.get('n',0)}）\n")
    ftest = regime.get("f_test", {})
    md.append(
        f"- **F 检验**（三状态 IC 分布差异）: F={fnum(ftest.get('f_stat',float('nan')))}，"
        f"p={fnum(ftest.get('p_value',float('nan')),6)} → "
        f"{'状态间差异显著' if ftest.get('significant_5pct') else '状态间差异不显著'}\n"
    )
    md.append(f"![各状态 RankIC](figures/{os.path.basename(paths['ic_regime'])})\n")
    if not shap_skipped:
        md.append(
            f"- **SHAP 全局特征重要性**: 采用 {shap_r['global']['method']} 方法。"
            "Top 特征见下图，模型并非依赖单一因子，而是多因子聚合决策。\n"
        )
        md.append(f"![Top 特征重要性](figures/{os.path.basename(paths['shap_top'])})\n")
    else:
        md.append(
            "- **SHAP 全局特征重要性**: 已跳过（神经网络模型无树结构，SHAP TreeExplainer 不适用；"
            "特征重要性可改用各子模型首层权重幅值近似）。\n"
        )

    md.append("### 1.3 投资组合检验（分组收益 / 多空组合 / 换手率）\n")
    gstat = port["group_stats"]
    gmr = port["grouping"]["group_mean_ret"]
    if gmr.is_monotonic_increasing:
        grad_txt = "呈单调上升梯度，验证因子有效性"
    elif gmr.is_monotonic_decreasing:
        grad_txt = "呈单调下降梯度（高分组收益更低，因子方向可能倒置，需关注）"
    else:
        grad_txt = "未呈严格单调梯度（存在组内倒挂，建议结合 RankIC 综合判断）"
    md.append(
        f"- **分组收益**: 十分位分组中头组（高分）平均日收益 "
        f"{fnum(gmr.iloc[-1]*100,3)}%，"
        f"尾组 {fnum(gmr.iloc[0]*100,3)}%，{grad_txt}。\n"
        f"- **多空组合**: 年化收益 {fnum(gstat.get('ann_return',float('nan'))*100,2)}%，"
        f"Sharpe {fnum(gstat.get('sharpe',float('nan')))}, "
        f"最大回撤 {fnum(gstat.get('max_drawdown',float('nan'))*100,2)}%\n"
    )
    md.append(f"![分组收益](figures/{os.path.basename(paths['grouping'])})\n")
    md.append(f"![多空累计净值](figures/{os.path.basename(paths['ls_cum'])})\n")
    md.append(
        f"- **换手率**: 多头腿日均换手 {fnum(port['turnover']['mean_turnover_long']*100,2)}%，"
        f"双边日均换手 {fnum(port['turnover']['mean_turnover_total']*100,2)}%\n"
    )
    md.append(f"![换手率](figures/{os.path.basename(paths['turnover'])})\n")

    md.append("### 1.4 交易成本检验（扣除成本后收益）\n")
    md.append(
        f"- 单边成本假设 {fnum(tcost['cost_per_trade']*100,3)}%（双边 {fnum(tcost['cost_per_trade']*200,3)}%）\n"
        f"- **扣费前**多空年化: {fnum(tcost['gross'].get('ann_return',float('nan'))*100,2)}%，"
        f"Sharpe {fnum(tcost['gross'].get('sharpe',float('nan')))}\n"
        f"- **扣费后**多空年化: {fnum(tcost['net'].get('ann_return',float('nan'))*100,2)}%，"
        f"Sharpe {fnum(tcost['net'].get('sharpe',float('nan')))}\n"
        f"- **成本侵蚀（年化）**: {fnum(tcost['cost_drag_ann']*100,2)}%\n"
    )

    md.append("## 二、SHAP 方法框架与全局特征重要性\n")
    if not shap_skipped:
        md.append(
            "SHAP（SHapley Additive exPlanations）基于博弈论将模型预测在特征间公平分配，"
            "全局重要性取各样本 |SHAP| 的均值。本报告对多目标模型的每个 LightGBM booster 分别做 "
            "TreeExplainer，再按目标权重聚合，得到整体特征贡献。\n"
        )
        md.append(f"![Top 特征重要性](figures/{os.path.basename(paths['shap_top'])})\n")
    else:
        md.append(
            "本模型为神经网络（无树结构），SHAP TreeExplainer 不适用，已跳过 SHAP 方法框架分析。"
            "如需神经网络的特征重要性，可改用各子模型首层权重的幅值近似（feature_importance）。\n"
        )

    md.append("## 三、财务 / 交易 / 估值 三类特征边际贡献解析\n")
    if not shap_skipped:
        share = cat["share"]
        md.append(
            f"- 财务类边际贡献占比: **{fnum(share.get('财务',0)*100,1)}%**\n"
            f"- 交易类边际贡献占比: **{fnum(share.get('交易',0)*100,1)}%**\n"
            f"- 估值类边际贡献占比: **{fnum(share.get('估值',0)*100,1)}%**\n"
            f"- 其他类占比: {fnum(share.get('其他',0)*100,1)}%\n"
        )
        md.append(f"![三类特征边际贡献](figures/{os.path.basename(paths['category'])})\n")
    else:
        md.append("- （SHAP 跳过，三类特征边际贡献不可用；如需可改用神经网络首层权重近似。）\n")

    md.append("## 四、机制分析\n")
    md.append("### 4.1 特征交互效应分析\n")
    if not shap_skipped:
        md.append(mech["interaction"] + "\n")
        md.append(f"![特征交互](figures/{os.path.basename(paths['interaction'])})\n")
    else:
        md.append("- （SHAP 交互分析已跳过；神经网络模型可用置换重要性或积分梯度近似。）\n")
    md.append("### 4.2 模型间 SHAP 一致性检验\n")
    if not shap_skipped:
        md.append(mech["consistency"] + "\n")
        md.append(f"![跨模型一致性](figures/{os.path.basename(paths['consistency'])})\n")
    else:
        md.append("- （SHAP 一致性检验已跳过。）\n")
    md.append("### 4.3 A 股特有定价规律经济学解释\n")
    md.append(mech["a_share_economics"] + "\n")

    md.append("## 五、关键指标速览（metrics.csv）\n")
    md.append("| 指标 | 数值 |\n|---|---|")
    for k, v in row.items():
        md.append(f"| {k} | {v} |")

    md_path = os.path.join(out_dir, "report.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return {"report_md": md_path, "metrics_csv": metrics_path, "figures": paths}
