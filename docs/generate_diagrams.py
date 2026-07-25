# -*- coding: utf-8 -*-
"""Generate focused architecture diagrams (PNG) for the system-overview chapter.
Each diagram is intentionally small in scope so it stays readable once inserted into a Word doc.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FONT = r"C:/Windows/Fonts/simhei.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

OUT = r"G:/quantified_decision/docs/figures"
os.makedirs(OUT, exist_ok=True)

# ---- palette ----
C_INFRA    = "#2E75B6"   # blue   - 基础设施
C_COMPUTE  = "#548235"   # green  - 计算核心
C_DECISION = "#C55A11"   # orange - 决策应用
C_INTERFACE= "#7030A0"   # purple - 接口编排
C_CONTRACT = "#BF9000"   # gold   - 共享契约
C_DATA     = "#1F8A70"   # teal   - 数据/存储
GREY       = "#595959"
LIGHT      = "#F2F2F2"

def new_fig(w_inch, h_inch):
    """Create figure with axes covering full area; coords in 'data' units."""
    fig, ax = plt.subplots(figsize=(w_inch, h_inch), dpi=180)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 700)
    ax.axis("off")
    return fig, ax

def box(ax, cx, cy, cw, ch, text, fc, tc="white", fs=12,
        weight="bold", ec=None, lw=1.5):
    ec = ec or fc
    rsize = min(10.0, cw * 0.08)
    p = FancyBboxPatch((cx, cy), cw, ch,
                       boxstyle=f"round,pad=0,rounding_size={rsize:.1f}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    # auto-wrap long text
    if len(text) > 14:
        mid = len(text) // 2
        best = mid
        for i in range(mid, 0, -1):
            if text[i] in " ·/\n":
                best = i + 1
                break
        t1, t2 = text[:best].strip(), text[best:].strip()
        if t2:
            ax.text(cx + cw/2, cy + ch*0.62, t1, ha="center", va="center",
                    color=tc, fontsize=fs, weight=weight, zorder=4)
            ax.text(cx + cw/2, cy + ch*0.32, t2, ha="center", va="center",
                    color=tc, fontsize=fs*0.9, weight=weight, zorder=4)
            return
    ax.text(cx + cw/2, cy + ch/2, text, ha="center", va="center",
            color=tc, fontsize=fs, weight=weight, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=GREY, lw=2.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle=style, mutation_scale=13,
                 linewidth=lw, color=color, linestyle=ls,
                 connectionstyle="arc3,rad=0", zorder=2))

def label_tab(ax, x, y, w, h, text, fc):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6",
                       linewidth=0, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color="white", fontsize=13, weight="bold", zorder=5)


# =====================================================================
# 图1：逻辑分层架构（宏观四层）
# =====================================================================
def fig1():
    fig, ax = new_fig(11, 7.6)
    layers = [
        (C_INTERFACE, "接口 / 编排层", [
            ("可视化 visualization", C_INTERFACE),
            ("编排 scripts · shell", C_INTERFACE),
            ("测试 tests", C_INTERFACE),
        ]),
        (C_DECISION, "决策应用层", [
            ("回测 backtest", C_DECISION),
            ("因子分析 analysis", C_DECISION),
            ("选股 · 实盘 automation", C_DECISION),
        ]),
        (C_COMPUTE, "计算核心层", [
            ("因子计算 factors", C_COMPUTE),
            ("ML 模型 models", C_COMPUTE),
            ("神经网络 neural", C_COMPUTE),
            ("技术基元 core/analysis", C_COMPUTE),
        ]),
        (C_INFRA, "基础设施层", [
            ("配置 config", C_INFRA),
            ("数据 ETL core/data", C_INFRA),
            ("存储 database", C_INFRA),
        ]),
    ]
    x0, xtab = 30, 140
    band_h = 110
    gap = 28
    y = 40
    for color, name, mods in layers:
        label_tab(ax, x0, y, xtab, band_h, name, color)
        n = len(mods)
        mw = 150
        sp = 18
        sx = x0 + xtab + 25
        for i, (mt, mc) in enumerate(mods):
            bx = sx + i * (mw + sp)
            box(ax, bx, y, mw, band_h, mt, mc, fs=12)
        y += band_h + gap
    # dependency arrow on right side
    arrow(ax, 950, 55, 950, y - gap - 15, color=C_CONTRACT, lw=2.8)
    ax.text(962, (y + 55)/2, "依赖方向 ↑\n(上层依赖下层)", color=C_CONTRACT,
            fontsize=12, weight="bold", va="center", ha="left")
    ax.set_title("图 1  系统逻辑分层架构（宏观视图）", fontsize=16, weight="bold",
                 color="#222222", pad=12, loc="left")
    fig.savefig(os.path.join(OUT, "fig1_layered_architecture.png"),
                bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("fig1 done")


# =====================================================================
# 图2：模块解耦与依赖关系（核心模块 + 共享契约高亮）
# =====================================================================
def fig2():
    fig, ax = new_fig(11.5, 8.4)
    # columns: infra | compute | decision | contracts
    # each column has stacked nodes
    col_x = [35, 280, 580, 850]
    node_w = [190, 230, 225, 135]
    node_h = 90
    dy = 105
    y_base = 45

    infra_nodes = [
        ("配置 config/\n(被普遍依赖的叶子)", C_INFRA),
        ("数据 ETL\ncore/data", C_INFRA),
        ("PIT 特征库\njydb_feature_store", C_DATA),
    ]
    compute_nodes = [
        ("因子计算器\nComprehensiveFactorCalculator", C_COMPUTE),
        ("ML 模型\nml_factor_model", C_COMPUTE),
        ("神经网络\ncore/neural", C_COMPUTE),
    ]
    decision_nodes = [
        ("回测引擎\ncore/backtest", C_DECISION),
        ("因子分析\nanalysis/", C_DECISION),
        ("选股 · 实盘\nautomation", C_DECISION),
    ]
    contract_nodes = [
        ("共享契约:\n因子缓存 .parquet", C_CONTRACT),
        ("共享契约:\nnorm_stats.pkl", C_CONTRACT),
        ("共享契约:\nexit_rules", C_CONTRACT),
    ]

    def draw_col(nodes, cx, nw):
        for i, (t, c) in enumerate(nodes):
            box(ax, cx, y_base + i*dy, nw, node_h, t, c, fs=11.5)

    draw_col(infra_nodes, col_x[0], node_w[0])
    draw_col(compute_nodes, col_x[1], node_w[1])
    draw_col(decision_nodes, col_x[2], node_w[2])
    draw_col(contract_nodes, col_x[3], node_w[3])

    # arrows: infra -> compute
    arrow(ax, col_x[1], y_base+dy*0.5+node_h/2, col_x[0]+node_w[0],
          y_base+dy*0.5+node_h/2, color=GREY)
    arrow(ax, col_x[1], y_base+dy*1.5+node_h/2, col_x[0]+node_w[0],
          y_base+dy*1.5+node_h/2, color=GREY)
    arrow(ax, col_x[1], y_base+dy*2.5+node_h/2, col_x[0]+node_w[0],
          y_base+dy*2.5+node_h/2, color=GREY)
    # compute -> decision
    arrow(ax, col_x[2], y_base+dy*0.5+node_h/2, col_x[1]+node_w[1],
          y_base+dy*0.5+node_h/2, color=GREY)
    arrow(ax, col_x[2], y_base+dy*1.5+node_h/2, col_x[1]+node_w[1],
          y_base+dy*1.5+node_h/2, color=GREY)
    arrow(ax, col_x[2], y_base+dy*2.5+node_h/2, col_x[1]+node_w[1],
          y_base+dy*2.5+node_h/2, color=GREY)
    # contracts consumed by decision
    arrow(ax, col_x[3], y_base+dy*0.5+node_h/2, col_x[2]+node_w[2],
          y_base+dy*0.5+node_h/2, color=C_CONTRACT)
    arrow(ax, col_x[3], y_base+dy*1.5+node_h/2, col_x[2]+node_w[2],
          y_base+dy*1.5+node_h/2, color=C_CONTRACT)
    arrow(ax, col_x[3], y_base+dy*2.5+node_h/2, col_x[2]+node_w[2],
          y_base+dy*2.5+node_h/2, color=C_CONTRACT)
    # compute produces contracts (dashed)
    arrow(ax, col_x[3], y_base+dy*0.5+node_h/2, col_x[1]+node_w[1],
          y_base+dy*0.5+node_h/2, color=C_CONTRACT, ls=(0,(5,3)))
    arrow(ax, col_x[3], y_base+dy*1.5+node_h/2, col_x[1]+node_w[1],
          y_base+dy*1.5+node_h/2, color=C_CONTRACT, ls=(0,(5,3)))
    arrow(ax, col_x[3], y_base+dy*2.5+node_h/2, col_x[1]+node_w[1],
          y_base+dy*2.5+node_h/2, color=C_CONTRACT, ls=(0,(5,3)))

    # legend
    ax.text(col_x[0], 385, "实线 = 模块依赖（上层→下层）", color=GREY, fontsize=11, weight="bold")
    ax.text(col_x[0], 360, "金线 = 通过共享契约解耦（接口一致、可替换）", color=C_CONTRACT, fontsize=11, weight="bold")

    ax.set_title("图 2  模块解耦与依赖关系", fontsize=16, weight="bold",
                 color="#222222", pad=12, loc="left")
    fig.savefig(os.path.join(OUT, "fig2_module_decoupling.png"),
                bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("fig2 done")


# =====================================================================
# 图3：解耦契约机制（四大共享契约 zoomed）
# =====================================================================
def fig3():
    fig, ax = new_fig(11.5, 6.8)
    contracts = [
        (C_DATA, "PIT 特征库",
         "available_date <= 交易日\n读取，从根上杜绝前视泄露",
         "数据层 -> 因子层"),
        (C_COMPUTE, "因子缓存 .parquet",
         "训练产出，回测/选股/分析\n共用同一份，因子零错位",
         "计算层 / 应用层"),
        (C_CONTRACT, "横截面归一化 norm_stats.pkl",
         "训练期统计量推理期复用，\n避免未来信息混入标准化",
         "计算层 -> 应用层"),
        (C_DECISION, "统一退出规则 exit_rules",
         "回测与实盘共用同一判定逻辑，\n研究态 = 生产态",
         "回测层 / 实盘层"),
    ]
    cw, ch = 210, 170
    body_h = 125
    x_start = 42
    y_top = 380
    gap = 28
    for i, (c, title, body, scope) in enumerate(contracts):
        cx = x_start + i * (cw + gap)
        box(ax, cx, y_top, cw, ch, title, c, fs=13)
        box(ax, cx, y_top - body_h - 12, cw, body_h, body, LIGHT,
            tc="#222222", fs=11, weight="normal", ec="#BBBBBB", lw=1.2)
        ax.text(cx + cw/2, y_top - body_h - 26, scope, ha="center", va="center",
                color=c, fontsize=10.5, weight="bold")
    ax.set_title("图 3  解耦契约：四大共享机制", fontsize=16, weight="bold",
                 color="#222222", pad=12, loc="left")
    fig.savefig(os.path.join(OUT, "fig3_shared_contracts.png"),
                bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("fig3 done")


# =====================================================================
# 图4：核心业务逻辑端到端数据流（pipeline）
# =====================================================================
def fig4():
    fig, ax = new_fig(11.5, 8.8)
    sw, sh = 155, 85
    sy = 480
    stages = [
        (35, "聚源 JYDB\n(SQL Server)", C_INFRA),
        (250, "Bronze\njydb_raw.db", C_INFRA),
        (465, "Silver\nfeatures/daily/meta", C_DATA),
        (680, "因子计算\n+ 特征工程", C_COMPUTE),
    ]
    for (x, t, c) in stages:
        box(ax, x, sy, sw, sh, t, c, fs=11.5)
    for i in range(len(stages)-1):
        arrow(ax, stages[i][0]+sw, sy+sh/2, stages[i+1][0], sy+sh/2, color=GREY)

    # vertical split from 因子计算
    mx = 680 + sw/2
    arrow(ax, mx, sy, mx, 365, color=GREY)
    box(ax, 680, 370, sw, 75, "多目标标签\n(forward-only 正交)", C_COMPUTE, fs=11)
    arrow(ax, mx, 370, mx, 275, color=GREY)
    box(ax, 680, 270, sw, 75, "ML 模型训练\nXGB / LGB", C_COMPUTE, fs=11)
    arrow(ax, mx, 270, mx, 175, color=GREY)
    box(ax, 680, 165, sw, 75, "模型 + norm_stats\n+ 因子缓存", C_CONTRACT, fs=11)

    # three branches consuming model/cache
    bx = 860
    consumers = [
        (bx, 480, "回测引擎\nbacktest", C_DECISION),
        (bx, 330, "每日选股 -> 实盘", C_DECISION),
        (bx, 165, "因子分析\nanalysis", C_DECISION),
    ]
    for (cx, cy, ct, cc) in consumers:
        box(ax, cx, cy, sw, 75, ct, cc, fs=11.5)
        arrow(ax, 680+sw, cy+37.5, bx, cy+37.5, color=C_CONTRACT)

    # neural branch reuses cache
    box(ax, 465, 260, sw, 75, "神经网络训练\n(复用缓存/标签/归一化)", C_COMPUTE, fs=10.5)
    arrow(ax, 465+sw/2, sy, 465+sw/2, 335, color=GREY)
    arrow(ax, 465+sw, 297.5, 680, 297.5, color=GREY)

    # anti-leak note
    box(ax, 35, 160, 145, 75, "防前视泄露\nPIT + t+1 标签", C_DATA, fs=11)
    arrow(ax, 35+145, 197.5, 107.5, 197.5, color=C_DATA)

    ax.text(35, 118, "说明：虚线路径表示\"复用既有产物\"；实线为主数据流。",
            color=GREY, fontsize=10.5, weight="bold")
    ax.set_title("图 4  核心业务逻辑：端到端数据流", fontsize=16, weight="bold",
                 color="#222222", pad=12, loc="left")
    fig.savefig(os.path.join(OUT, "fig4_data_pipeline.png"),
                bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("fig4 done")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("ALL DONE ->", OUT)
