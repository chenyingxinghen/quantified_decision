"""
K线形态参数可视化 — 形态正确性检验

每种形态绘制：
  左：当前参数下的"典型"K线（恰好在边界上）
  右：参数说明 + 边界值 + 若调大/调小参数会如何影响识别

运行：python visualize_candlestick_params.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from config.factor_config import FactorConfig as C

# ─────────────────────────────────────────────────────────────────────────────
# 基础绘图工具
# ─────────────────────────────────────────────────────────────────────────────

def draw_candle(ax, x, open_, high, low, close, width=0.4,
                color_bull='#26a69a', color_bear='#ef5350', color_doji='#888888'):
    """在 ax 的 x 位置画一根K线"""
    is_bull = close >= open_
    is_doji = abs(close - open_) < 1e-9
    color = color_doji if is_doji else (color_bull if is_bull else color_bear)
    # 影线
    ax.plot([x, x], [low, high], color=color, lw=1.5, zorder=2)
    # 实体
    body_lo = min(open_, close)
    body_hi = max(open_, close)
    if body_hi - body_lo < 1e-9:
        ax.plot([x - width/2, x + width/2], [body_lo, body_lo], color=color, lw=2, zorder=3)
    else:
        rect = mpatches.FancyBboxPatch(
            (x - width/2, body_lo), width, body_hi - body_lo,
            boxstyle='square,pad=0', facecolor=color, edgecolor=color, zorder=3
        )
        ax.add_patch(rect)


def annotate_dim(ax, x0, x1, y, label, color='#555', fontsize=7.5, above=True):
    """双向箭头标注尺寸"""
    dy = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ya = y + dy if above else y - dy
    ax.annotate('', xy=(x1, ya), xytext=(x0, ya),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.2))
    ax.text((x0+x1)/2, ya + (dy*0.8 if above else -dy*1.6),
            label, ha='center', va='bottom' if above else 'top',
            fontsize=fontsize, color=color)


def param_box(ax, lines, title='参数说明', fontsize=8):
    """在 ax 上绘制参数说明文本框"""
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.05, 0.97, title, fontsize=9, fontweight='bold', va='top',
            transform=ax.transAxes, color='#222')
    for i, (txt, color) in enumerate(lines):
        ax.text(0.05, 0.88 - i * 0.115, txt, fontsize=fontsize, va='top',
                transform=ax.transAxes, color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8',
                          edgecolor='#ddd', alpha=0.9) if color != '#555' else None)


def setup_candle_ax(ax, title, ylim=None):
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.set_xlim(-0.8, 0.8)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# 每种形态的绘制函数
# ─────────────────────────────────────────────────────────────────────────────

def fig_single_candles():
    """单根K线形态：十字星、锤子线、上吊线、射击之星、倒锤线、光头光脚、纺锤线"""
    fig, axes = plt.subplots(2, 7, figsize=(22, 9),
                             gridspec_kw={'height_ratios': [3, 1.8]})
    fig.suptitle('单根K线形态 — 当前参数下的典型形态', fontsize=13, fontweight='bold', y=1.01)

    price = 10.0  # 基准价格

    # ── 1. 十字星 ──────────────────────────────────────────────────────────
    ax = axes[0][0]
    body = C.DOJI_THRESHOLD * price * 0.8   # 恰好在阈值内
    o, c = price - body/2, price + body/2
    h, l = price + price*0.015, price - price*0.015
    draw_candle(ax, 0, o, h, l, c)
    setup_candle_ax(ax, '十字星 (Doji)', ylim=(l*0.998, h*1.002))
    ax.axhline(o, color='#aaa', lw=0.8, ls=':')
    ax.axhline(c, color='#aaa', lw=0.8, ls=':')
    annotate_dim(ax, -0.25, 0.25, (o+c)/2,
                 f'实体={body/price:.4f}p\n< DOJI={C.DOJI_THRESHOLD}', '#DD4444')

    param_box(axes[1][0], [
        (f'DOJI_THRESHOLD = {C.DOJI_THRESHOLD}', '#DD4444'),
        (f'实体/收盘价 < {C.DOJI_THRESHOLD}', '#555'),
        ('↑调大 → 更多K线被识别为十字星', '#E65100'),
        ('↓调小 → 只识别极细实体', '#1565C0'),
    ], '十字星参数')

    # ── 2. 锤子线 ──────────────────────────────────────────────────────────
    ax = axes[0][1]
    body = price * 0.012
    o, c = price, price + body          # 阳线锤子
    lower = body * (C.HAMMER_LOWER_SHADOW_RATIO + 0.1)  # 下影略超阈值
    upper = body * (C.HAMMER_UPPER_SHADOW_RATIO - 0.05) # 上影略低于阈值
    h = c + upper
    l = o - lower
    draw_candle(ax, 0, o, h, l, c)
    setup_candle_ax(ax, '锤子线 (Hammer)', ylim=(l*0.997, h*1.003))
    annotate_dim(ax, 0.22, 0.22, l, f'下影={lower/body:.1f}×实体\n≥ {C.HAMMER_LOWER_SHADOW_RATIO}', '#DD4444', above=False)
    annotate_dim(ax, -0.22, 0.22, c, f'上影={upper/body:.2f}×实体\n≤ {C.HAMMER_UPPER_SHADOW_RATIO}', '#1565C0')

    param_box(axes[1][1], [
        (f'LOWER_RATIO = {C.HAMMER_LOWER_SHADOW_RATIO}  (下影/实体 ≥)', '#DD4444'),
        (f'UPPER_RATIO = {C.HAMMER_UPPER_SHADOW_RATIO}  (上影/实体 ≤)', '#1565C0'),
        ('↑LOWER → 要求更长下影，更严格', '#E65100'),
        ('↑UPPER → 允许更长上影，更宽松', '#E65100'),
    ], '锤子线参数')

    # ── 3. 上吊线 ──────────────────────────────────────────────────────────
    ax = axes[0][2]
    body = price * 0.012
    o, c = price + body, price          # 阴线上吊
    lower = body * (C.HAMMER_LOWER_SHADOW_RATIO + 0.1)
    upper = body * (C.HAMMER_UPPER_SHADOW_RATIO - 0.05)
    h = o + upper
    l = c - lower
    draw_candle(ax, 0, o, h, l, c)
    setup_candle_ax(ax, '上吊线 (Hanging Man)', ylim=(l*0.997, h*1.003))
    annotate_dim(ax, 0.22, 0.22, l, f'下影≥{C.HAMMER_LOWER_SHADOW_RATIO}×实体', '#DD4444', above=False)
    annotate_dim(ax, -0.22, 0.22, o, f'上影≤{C.HAMMER_UPPER_SHADOW_RATIO}×实体', '#1565C0')

    param_box(axes[1][2], [
        (f'同锤子线参数（形状相同）', '#555'),
        (f'LOWER_RATIO = {C.HAMMER_LOWER_SHADOW_RATIO}', '#DD4444'),
        (f'UPPER_RATIO = {C.HAMMER_UPPER_SHADOW_RATIO}', '#1565C0'),
        ('区别：出现在高位 price_pos > ' + str(C.PRICE_POS_HIGH), '#9C27B0'),
    ], '上吊线参数')

    # ── 4. 射击之星 ────────────────────────────────────────────────────────
    ax = axes[0][3]
    body = price * 0.010
    o, c = price, price - body          # 阴线
    upper = body * (C.SHOOTING_STAR_UPPER_RATIO + 0.2)
    lower = body * (C.SHOOTING_STAR_LOWER_RATIO - 0.05)
    h = o + upper
    l = c - lower
    draw_candle(ax, 0, o, h, l, c)
    setup_candle_ax(ax, '射击之星 (Shooting Star)', ylim=(l*0.997, h*1.003))
    annotate_dim(ax, -0.22, 0.22, o, f'上影={upper/body:.1f}×实体\n≥ {C.SHOOTING_STAR_UPPER_RATIO}', '#DD4444')
    annotate_dim(ax, 0.22, 0.22, c, f'下影≤{C.SHOOTING_STAR_LOWER_RATIO}×实体', '#1565C0', above=False)

    param_box(axes[1][3], [
        (f'UPPER_RATIO = {C.SHOOTING_STAR_UPPER_RATIO}  (上影/实体 ≥)', '#DD4444'),
        (f'LOWER_RATIO = {C.SHOOTING_STAR_LOWER_RATIO}  (下影/实体 ≤)', '#1565C0'),
        ('↑UPPER → 要求更长上影，更严格', '#E65100'),
        ('出现在高位 price_pos > ' + str(C.PRICE_POS_HIGH), '#9C27B0'),
    ], '射击之星参数')

    # ── 5. 倒锤线 ──────────────────────────────────────────────────────────
    ax = axes[0][4]
    body = price * 0.010
    o, c = price, price + body          # 阳线
    upper = body * (C.SHOOTING_STAR_UPPER_RATIO + 0.2)
    lower = body * (C.SHOOTING_STAR_LOWER_RATIO - 0.05)
    h = c + upper
    l = o - lower
    draw_candle(ax, 0, o, h, l, c)
    setup_candle_ax(ax, '倒锤线 (Inverted Hammer)', ylim=(l*0.997, h*1.003))
    annotate_dim(ax, -0.22, 0.22, c, f'上影≥{C.SHOOTING_STAR_UPPER_RATIO}×实体', '#DD4444')
    annotate_dim(ax, 0.22, 0.22, o, f'下影≤{C.SHOOTING_STAR_LOWER_RATIO}×实体', '#1565C0', above=False)

    param_box(axes[1][4], [
        (f'同射击之星参数（形状相同）', '#555'),
        (f'UPPER_RATIO = {C.SHOOTING_STAR_UPPER_RATIO}', '#DD4444'),
        (f'LOWER_RATIO = {C.SHOOTING_STAR_LOWER_RATIO}', '#1565C0'),
        ('区别：出现在低位 price_pos < ' + str(C.PRICE_POS_LOW_STRICT), '#9C27B0'),
    ], '倒锤线参数')

    # ── 6. 光头光脚 ────────────────────────────────────────────────────────
    ax = axes[0][5]
    body = price * (C.MARUBOZU_MIN_BODY_RATIO + 0.005)
    o = price - body/2
    c = price + body/2
    shadow = price * C.MARUBOZU_SHADOW_RATIO * 0.5   # 影线极短，在阈值内
    h = c + shadow
    l = o - shadow
    draw_candle(ax, 0, o, h, l, c)
    setup_candle_ax(ax, '光头光脚 (Marubozu)', ylim=(l*0.997, h*1.003))
    annotate_dim(ax, -0.25, 0.25, (o+c)/2,
                 f'实体={body/price:.3f}p\n≥ MIN={C.MARUBOZU_MIN_BODY_RATIO}', '#DD4444')
    ax.text(0.35, h, f'影线<{C.MARUBOZU_SHADOW_RATIO}p', fontsize=7, color='#1565C0', va='bottom')
    ax.text(0.35, l, f'影线<{C.MARUBOZU_SHADOW_RATIO}p', fontsize=7, color='#1565C0', va='top')

    param_box(axes[1][5], [
        (f'MIN_BODY = {C.MARUBOZU_MIN_BODY_RATIO}  (实体/收盘价 ≥)', '#DD4444'),
        (f'SHADOW_RATIO = {C.MARUBOZU_SHADOW_RATIO}  (影线/收盘价 ≤)', '#1565C0'),
        ('↑MIN_BODY → 只识别大阳/阴线', '#E65100'),
        ('↑SHADOW → 允许更长影线', '#E65100'),
    ], '光头光脚参数')

    # ── 7. 纺锤线 ──────────────────────────────────────────────────────────
    ax = axes[0][6]
    candle_range = price * 0.03
    body = candle_range * (C.SPINNING_TOP_BODY_RATIO - 0.01)  # 实体刚好在阈值内
    o = price - body/2
    c = price + body/2
    remaining = candle_range - body
    upper = remaining * 0.52   # 上下影线近似对称
    lower = remaining * 0.48
    h = c + upper
    l = o - lower
    draw_candle(ax, 0, o, h, l, c)
    setup_candle_ax(ax, '纺锤线 (Spinning Top)', ylim=(l*0.997, h*1.003))
    annotate_dim(ax, -0.25, 0.25, (o+c)/2,
                 f'实体/全幅={body/candle_range:.2f}\n< {C.SPINNING_TOP_BODY_RATIO}', '#DD4444')

    param_box(axes[1][6], [
        (f'BODY_RATIO = {C.SPINNING_TOP_BODY_RATIO}  (实体/全幅 ≤)', '#DD4444'),
        (f'SYMMETRY = {C.SPINNING_TOP_SHADOW_SYMMETRY}  (|上-下|/全幅 ≤)', '#1565C0'),
        ('↑BODY_RATIO → 允许更大实体', '#E65100'),
        ('↑SYMMETRY → 允许更不对称影线', '#E65100'),
    ], '纺锤线参数')

    fig.tight_layout(h_pad=0.5)
    return fig


def fig_double_candles():
    """双根K线形态：看涨吞没、看跌吞没、刺穿线、乌云盖顶、孕线"""
    fig, axes = plt.subplots(2, 5, figsize=(18, 9),
                             gridspec_kw={'height_ratios': [3, 1.8]})
    fig.suptitle('双根K线形态 — 当前参数下的典型形态', fontsize=13, fontweight='bold', y=1.01)

    price = 10.0

    # ── 1. 看涨吞没 ────────────────────────────────────────────────────────
    ax = axes[0][0]
    prev_body = price * 0.018
    po, pc = price + prev_body/2, price - prev_body/2   # 前根阴线
    curr_body = prev_body * (1 + C.ENGULFING_SIGNIFICANCE + 0.01)
    co = pc - curr_body * 0.1
    cc = co + curr_body                                  # 当根阳线，完全吞没
    ph = po + price*0.005; pl = pc - price*0.005
    ch = cc + price*0.004; cl = co - price*0.004
    draw_candle(ax, -0.3, po, ph, pl, pc)
    draw_candle(ax, 0.3, co, ch, cl, cc)
    setup_candle_ax(ax, '看涨吞没 (Bullish Engulfing)',
                    ylim=(min(pl,cl)*0.997, max(ph,ch)*1.003))
    ax.annotate('', xy=(0.3-0.2, cc), xytext=(-0.3+0.2, po),
                arrowprops=dict(arrowstyle='->', color='#26a69a', lw=1.5))
    ax.text(0, (cc+po)/2 + price*0.003, f'cc > po×{1+C.ENGULFING_SIGNIFICANCE:.3f}',
            ha='center', fontsize=7, color='#DD4444')

    param_box(axes[1][0], [
        (f'SIGNIFICANCE = {C.ENGULFING_SIGNIFICANCE}', '#DD4444'),
        (f'当根收盘 > 前根开盘×{1+C.ENGULFING_SIGNIFICANCE:.3f}', '#555'),
        ('↑调大 → 要求更显著的吞没', '#E65100'),
        (f'低位触发: price_pos < {C.PRICE_POS_LOW_ENGULF}', '#9C27B0'),
    ], '看涨吞没参数')

    # ── 2. 看跌吞没 ────────────────────────────────────────────────────────
    ax = axes[0][1]
    prev_body = price * 0.018
    po, pc = price - prev_body/2, price + prev_body/2   # 前根阳线
    curr_body = prev_body * (1 + C.ENGULFING_SIGNIFICANCE + 0.01)
    co = pc + curr_body * 0.1
    cc = co - curr_body                                  # 当根阴线
    ph = pc + price*0.005; pl = po - price*0.005
    ch = co + price*0.004; cl = cc - price*0.004
    draw_candle(ax, -0.3, po, ph, pl, pc)
    draw_candle(ax, 0.3, co, ch, cl, cc)
    setup_candle_ax(ax, '看跌吞没 (Bearish Engulfing)',
                    ylim=(min(pl,cl)*0.997, max(ph,ch)*1.003))
    ax.text(0, (co+pc)/2 + price*0.003, f'cc < po×{1-C.ENGULFING_SIGNIFICANCE:.3f}',
            ha='center', fontsize=7, color='#DD4444')

    param_box(axes[1][1], [
        (f'SIGNIFICANCE = {C.ENGULFING_SIGNIFICANCE}', '#DD4444'),
        (f'当根收盘 < 前根开盘×{1-C.ENGULFING_SIGNIFICANCE:.3f}', '#555'),
        ('↑调大 → 要求更显著的吞没', '#E65100'),
        (f'高位触发: price_pos > {C.PRICE_POS_HIGH_ENGULF}', '#9C27B0'),
    ], '看跌吞没参数')

    # ── 3. 刺穿线 ──────────────────────────────────────────────────────────
    ax = axes[0][2]
    prev_body = price * 0.022
    po, pc = price + prev_body/2, price - prev_body/2   # 前根大阴线
    midpoint = (po + pc) / 2
    co = pc - price*0.005                               # 低开
    cc = midpoint + price*0.003                         # 收在中点以上但未超前开
    ph = po + price*0.004; pl = pc - price*0.008
    ch = cc + price*0.003; cl = co - price*0.003
    draw_candle(ax, -0.3, po, ph, pl, pc)
    draw_candle(ax, 0.3, co, ch, cl, cc)
    setup_candle_ax(ax, '刺穿线 (Piercing Line)',
                    ylim=(min(pl,cl)*0.997, max(ph,ch)*1.003))
    ax.axhline(midpoint, color='#FF8800', lw=1, ls='--', alpha=0.8)
    ax.text(0.55, midpoint, '中点', fontsize=7, color='#FF8800', va='center')
    ax.text(0.55, po, '前开', fontsize=7, color='#555', va='center')
    ax.annotate('', xy=(0.3+0.18, cc), xytext=(0.3+0.18, midpoint),
                arrowprops=dict(arrowstyle='<->', color='#26a69a', lw=1.2))
    ax.text(0.72, (cc+midpoint)/2, '收在\n中点上', fontsize=6.5, color='#26a69a', va='center')

    param_box(axes[1][2], [
        ('收盘 > 前根中点，< 前根开盘', '#555'),
        (f'低位触发: price_pos < {C.PRICE_POS_LOW_ENGULF}', '#9C27B0'),
        ('无独立比例参数', '#888'),
        ('↑LOW_ENGULF → 更宽松的低位判断', '#E65100'),
    ], '刺穿线参数')

    # ── 4. 乌云盖顶 ────────────────────────────────────────────────────────
    ax = axes[0][3]
    prev_body = price * 0.022
    po, pc = price - prev_body/2, price + prev_body/2   # 前根大阳线
    midpoint = (po + pc) / 2
    co = pc + price*0.005                               # 高开
    cc = midpoint - price*0.003                         # 收在中点以下
    ph = pc + price*0.008; pl = po - price*0.004
    ch = co + price*0.003; cl = cc - price*0.003
    draw_candle(ax, -0.3, po, ph, pl, pc)
    draw_candle(ax, 0.3, co, ch, cl, cc)
    setup_candle_ax(ax, '乌云盖顶 (Dark Cloud Cover)',
                    ylim=(min(pl,cl)*0.997, max(ph,ch)*1.003))
    ax.axhline(midpoint, color='#FF8800', lw=1, ls='--', alpha=0.8)
    ax.text(0.55, midpoint, '中点', fontsize=7, color='#FF8800', va='center')
    ax.annotate('', xy=(0.3+0.18, cc), xytext=(0.3+0.18, midpoint),
                arrowprops=dict(arrowstyle='<->', color='#ef5350', lw=1.2))
    ax.text(0.72, (cc+midpoint)/2, '收在\n中点下', fontsize=6.5, color='#ef5350', va='center')

    param_box(axes[1][3], [
        ('收盘 < 前根中点，> 前根开盘', '#555'),
        (f'高位触发: price_pos > {C.PRICE_POS_HIGH_ENGULF}', '#9C27B0'),
        ('无独立比例参数', '#888'),
        ('↑HIGH_ENGULF → 更严格的高位判断', '#E65100'),
    ], '乌云盖顶参数')

    # ── 5. 孕线 ────────────────────────────────────────────────────────────
    ax = axes[0][4]
    outer_body = price * 0.025
    inner_body = outer_body / (C.HARAMI_BODY_RATIO + 0.2)  # 内包实体明显更小
    po, pc = price + outer_body/2, price - outer_body/2    # 外包阴线
    io = price + inner_body/2
    ic = price - inner_body/2                               # 内包阳线
    ph = po + price*0.004; pl = pc - price*0.004
    ih = io + price*0.002; il = ic - price*0.002
    draw_candle(ax, -0.3, po, ph, pl, pc)
    draw_candle(ax, 0.3, io, ih, il, ic)
    setup_candle_ax(ax, '孕线 (Harami)',
                    ylim=(min(pl,il)*0.997, max(ph,ih)*1.003))
    annotate_dim(ax, -0.5, -0.5, pc, f'外包实体\n={outer_body/price:.3f}p', '#DD4444', above=False)
    annotate_dim(ax, 0.5, 0.5, ic, f'内包实体\n={inner_body/price:.3f}p', '#1565C0', above=False)
    ax.text(0, price + price*0.005,
            f'外/内 = {outer_body/inner_body:.1f} ≥ {C.HARAMI_BODY_RATIO}',
            ha='center', fontsize=7.5, color='#DD4444')

    param_box(axes[1][4], [
        (f'HARAMI_BODY_RATIO = {C.HARAMI_BODY_RATIO}', '#DD4444'),
        (f'外包实体 ≥ {C.HARAMI_BODY_RATIO}× 内包实体', '#555'),
        ('↑调大 → 要求外包更大，更严格', '#E65100'),
        (f'极端位置: pos<{C.PRICE_POS_LOW} 或 >{C.PRICE_POS_HIGH_STRICT}', '#9C27B0'),
    ], '孕线参数')

    fig.tight_layout(h_pad=0.5)
    return fig


def fig_triple_candles():
    """三根K线形态：晨星、暮星、三白兵、三乌鸦"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 9),
                             gridspec_kw={'height_ratios': [3, 1.8]})
    fig.suptitle('三根K线形态 — 当前参数下的典型形态', fontsize=13, fontweight='bold', y=1.01)

    price = 10.0

    # ── 1. 晨星 ────────────────────────────────────────────────────────────
    ax = axes[0][0]
    # 第一根：大阴线
    b1 = price * 0.022
    o1, c1 = price + b1/2, price - b1/2
    # 第二根：小实体（< STAR_SECOND_BODY_RATIO × b1）
    b2 = b1 * (C.STAR_SECOND_BODY_RATIO - 0.05)
    gap_down = price * 0.005
    o2 = c1 - gap_down
    c2 = o2 + b2
    # 第三根：大阳线，收在第一根中点以上
    mid1 = (o1 + c1) / 2
    b3 = price * 0.020
    o3 = c2 + price*0.003
    c3 = o3 + b3
    xs = [-0.5, 0, 0.5]
    candles = [(o1, o1+price*0.004, c1-price*0.004, c1),
               (o2, o2+price*0.002, o2-price*0.002, c2),
               (o3, c3+price*0.004, o3-price*0.003, c3)]
    for x, (o,h,l,c) in zip(xs, candles):
        draw_candle(ax, x, o, h, l, c, width=0.3)
    setup_candle_ax(ax, '晨星 (Morning Star)',
                    ylim=(min(c1,o2)*0.996, max(o1,c3)*1.004))
    ax.axhline(mid1, color='#FF8800', lw=1, ls='--', alpha=0.7)
    ax.text(0.65, mid1, '第1根\n中点', fontsize=6.5, color='#FF8800', va='center')
    ax.text(0, o2 - price*0.008,
            f'第2根实体={b2/b1:.2f}×第1根\n< {C.STAR_SECOND_BODY_RATIO}',
            ha='center', fontsize=7, color='#DD4444')

    param_box(axes[1][0], [
        (f'STAR_SECOND_BODY = {C.STAR_SECOND_BODY_RATIO}', '#DD4444'),
        (f'第2根实体 < {C.STAR_SECOND_BODY_RATIO}× 第1根实体', '#555'),
        ('第3根收盘 > 第1根中点', '#555'),
        (f'低位触发: price_pos < {C.PRICE_POS_LOW}', '#9C27B0'),
    ], '晨星参数')

    # ── 2. 暮星 ────────────────────────────────────────────────────────────
    ax = axes[0][1]
    b1 = price * 0.022
    o1, c1 = price - b1/2, price + b1/2   # 大阳线
    b2 = b1 * (C.STAR_SECOND_BODY_RATIO - 0.05)
    gap_up = price * 0.005
    o2 = c1 + gap_up
    c2 = o2 + b2
    mid1 = (o1 + c1) / 2
    b3 = price * 0.020
    o3 = c2 - price*0.003
    c3 = o3 - b3
    candles = [(o1, c1+price*0.004, o1-price*0.004, c1),
               (o2, c2+price*0.002, o2-price*0.002, c2),
               (o3, o3+price*0.003, c3-price*0.004, c3)]
    for x, (o,h,l,c) in zip(xs, candles):
        draw_candle(ax, x, o, h, l, c, width=0.3)
    setup_candle_ax(ax, '暮星 (Evening Star)',
                    ylim=(min(o1,c3)*0.996, max(c1,c2)*1.004))
    ax.axhline(mid1, color='#FF8800', lw=1, ls='--', alpha=0.7)
    ax.text(0.65, mid1, '第1根\n中点', fontsize=6.5, color='#FF8800', va='center')
    ax.text(0, c2 + price*0.006,
            f'第2根实体={b2/b1:.2f}×第1根\n< {C.STAR_SECOND_BODY_RATIO}',
            ha='center', fontsize=7, color='#DD4444')

    param_box(axes[1][1], [
        (f'STAR_SECOND_BODY = {C.STAR_SECOND_BODY_RATIO}', '#DD4444'),
        (f'第2根实体 < {C.STAR_SECOND_BODY_RATIO}× 第1根实体', '#555'),
        ('第3根收盘 < 第1根中点', '#555'),
        (f'高位触发: price_pos > {C.PRICE_POS_HIGH_STRICT}', '#9C27B0'),
    ], '暮星参数')

    # ── 3. 三白兵 ──────────────────────────────────────────────────────────
    ax = axes[0][2]
    step = price * 0.018
    bases = [price - step, price, price + step]
    for i, (x, base) in enumerate(zip(xs, bases)):
        o = base
        c = base + step * 0.9
        h = c + step * 0.1
        l = o - step * 0.05
        draw_candle(ax, x, o, h, l, c, width=0.3)
    setup_candle_ax(ax, '三白兵 (Three White Soldiers)',
                    ylim=(bases[0]-step*0.3, bases[2]+step*1.3))
    ax.annotate('', xy=(0.5, bases[2]+step*0.9), xytext=(-0.5, bases[0]+step*0.9),
                arrowprops=dict(arrowstyle='->', color='#26a69a', lw=2))
    ax.text(0, bases[1]+step*1.15, '连续三根阳线，逐步抬高',
            ha='center', fontsize=7.5, color='#26a69a')

    param_box(axes[1][2], [
        ('三根阳线 + 收盘逐步抬高', '#555'),
        (f'SOLDIERS_CROWS = {C.PRICE_POS_SOLDIERS_CROWS}', '#DD4444'),
        (f'触发条件: price_pos < {C.PRICE_POS_SOLDIERS_CROWS}', '#9C27B0'),
        ('↑调大 → 允许在更高位置触发', '#E65100'),
    ], '三白兵参数')

    # ── 4. 三乌鸦 ──────────────────────────────────────────────────────────
    ax = axes[0][3]
    step = price * 0.018
    bases = [price + step, price, price - step]
    for i, (x, base) in enumerate(zip(xs, bases)):
        o = base
        c = base - step * 0.9
        h = o + step * 0.05
        l = c - step * 0.1
        draw_candle(ax, x, o, h, l, c, width=0.3)
    setup_candle_ax(ax, '三乌鸦 (Three Black Crows)',
                    ylim=(bases[2]-step*1.3, bases[0]+step*0.3))
    ax.annotate('', xy=(0.5, bases[2]-step*0.9), xytext=(-0.5, bases[0]-step*0.9),
                arrowprops=dict(arrowstyle='->', color='#ef5350', lw=2))
    ax.text(0, bases[1]-step*1.15, '连续三根阴线，逐步下跌',
            ha='center', fontsize=7.5, color='#ef5350')

    param_box(axes[1][3], [
        ('三根阴线 + 收盘逐步下跌', '#555'),
        (f'SOLDIERS_CROWS = {C.PRICE_POS_SOLDIERS_CROWS}', '#DD4444'),
        (f'触发条件: price_pos > {C.PRICE_POS_SOLDIERS_CROWS}', '#9C27B0'),
        ('↓调小 → 允许在更低位置触发', '#E65100'),
    ], '三乌鸦参数')

    fig.tight_layout(h_pad=0.5)
    return fig


def fig_price_pos():
    """价格位置阈值示意图：用一段价格序列展示各阈值的含义"""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('价格位置阈值 (price_pos) — 各形态触发区间', fontsize=13, fontweight='bold')

    # 模拟一段价格走势
    np.random.seed(0)
    n = 60
    prices = 10 + np.cumsum(np.random.randn(n) * 0.08)
    lo = np.array([min(prices[max(0,i-C.CONTEXT_WINDOW):i+1]) for i in range(n)])
    hi = np.array([max(prices[max(0,i-C.CONTEXT_WINDOW):i+1]) for i in range(n)])
    pos = (prices - lo) / np.where(hi - lo < 1e-6, 1e-6, hi - lo)

    # 绘制价格
    ax2 = ax.twinx()
    ax2.plot(prices, color='#4C72B0', lw=1.5, alpha=0.6, label='价格')
    ax2.fill_between(range(n), lo, hi, alpha=0.08, color='#4C72B0')
    ax2.set_ylabel('价格', fontsize=9, color='#4C72B0')
    ax2.tick_params(labelsize=7)

    # 绘制 price_pos
    ax.plot(pos, color='#333', lw=2, label='price_pos')
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel('price_pos (0=低位, 1=高位)', fontsize=9)
    ax.set_xlabel(f'交易日（滚动窗口={C.CONTEXT_WINDOW}天）', fontsize=9)
    ax.tick_params(labelsize=7)

    # 阈值区间着色
    thresholds = [
        (C.PRICE_POS_LOW_STRICT,  '#9C27B0', '倒锤线低位', 'bottom'),
        (C.PRICE_POS_LOW,         '#26a69a', '锤子/晨星低位', 'bottom'),
        (C.PRICE_POS_LOW_ENGULF,  '#FF8800', '吞没/刺穿低位', 'bottom'),
        (C.PRICE_POS_SOLDIERS_CROWS, '#888', '三白兵/三乌鸦分界', None),
        (C.PRICE_POS_HIGH_ENGULF, '#FF8800', '吞没/乌云高位', 'top'),
        (C.PRICE_POS_HIGH_STRICT, '#26a69a', '暮星高位', 'top'),
        (C.PRICE_POS_HIGH,        '#9C27B0', '射击之星/上吊高位', 'top'),
    ]

    colors_low  = ['#9C27B020', '#26a69a20', '#FF880020']
    colors_high = ['#FF880020', '#26a69a20', '#9C27B020']

    low_vals  = sorted([t[0] for t in thresholds if t[3]=='bottom'])
    high_vals = sorted([t[0] for t in thresholds if t[3]=='top'])

    for i, v in enumerate(low_vals):
        ax.axhspan(0, v, alpha=0.25, color=['#9C27B0','#26a69a','#FF8800'][i], zorder=0)
    for i, v in enumerate(reversed(high_vals)):
        ax.axhspan(v, 1, alpha=0.25, color=['#9C27B0','#26a69a','#FF8800'][i], zorder=0)

    for val, color, label, side in thresholds:
        ax.axhline(val, color=color, lw=1.5, ls='--', alpha=0.9)
        xpos = 0.5 if side is None else (0.02 if side == 'bottom' else 0.98)
        ha = 'center' if side is None else ('left' if side == 'bottom' else 'right')
        ax.text(n * (0.5 if side is None else (0.02 if side=='bottom' else 0.98)),
                val + 0.015, f'{label} = {val}',
                fontsize=7.5, color=color, ha=ha, va='bottom')

    # 标注触发点
    for i in range(n):
        if pos[i] < C.PRICE_POS_LOW_STRICT:
            ax.scatter(i, pos[i], color='#9C27B0', s=30, zorder=5)
        elif pos[i] > C.PRICE_POS_HIGH:
            ax.scatter(i, pos[i], color='#9C27B0', s=30, zorder=5, marker='v')

    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # 图例说明
    legend_items = [
        mpatches.Patch(color='#9C27B0', alpha=0.5, label=f'严格低/高位 (倒锤/射击之星/上吊)'),
        mpatches.Patch(color='#26a69a', alpha=0.5, label=f'标准低/高位 (锤子/晨星/暮星)'),
        mpatches.Patch(color='#FF8800', alpha=0.5, label=f'宽松低/高位 (吞没/刺穿/乌云)'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=8)

    fig.tight_layout()
    return fig


def fig_param_impact():
    """参数影响对比：同一形态，不同参数值下的边界K线对比"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                             gridspec_kw={'height_ratios': [3, 1.5]})
    fig.suptitle('参数调整影响对比 — 边界K线示意', fontsize=13, fontweight='bold', y=1.01)

    price = 10.0

    # ── 锤子线：LOWER_RATIO 对比 ──────────────────────────────────────────
    for col, ratio in enumerate([1.0, C.HAMMER_LOWER_SHADOW_RATIO, 2.5]):
        ax = axes[0][col]
        body = price * 0.012
        o, c = price, price + body
        lower = body * ratio
        upper = body * 0.3
        h = c + upper; l = o - lower
        draw_candle(ax, 0, o, h, l, c)
        setup_candle_ax(ax, f'锤子线 LOWER={ratio}', ylim=(l*0.997, h*1.003))
        annotate_dim(ax, 0.22, 0.22, l, f'下影={ratio}×实体', '#DD4444', above=False)
        is_valid = ratio >= C.HAMMER_LOWER_SHADOW_RATIO
        status = '✓ 触发' if is_valid else '✗ 不触发'
        color = '#26a69a' if is_valid else '#ef5350'
        ax.text(0, h + price*0.002, status, ha='center', fontsize=9,
                fontweight='bold', color=color)

    param_box(axes[1][0], [
        (f'当前 LOWER_RATIO = {C.HAMMER_LOWER_SHADOW_RATIO}', '#DD4444'),
        ('左图(1.0): 下影不足 → 不触发', '#ef5350'),
        (f'中图({C.HAMMER_LOWER_SHADOW_RATIO}): 恰好在边界 → 触发', '#26a69a'),
        ('右图(2.5): 超过阈值 → 触发', '#26a69a'),
    ], 'LOWER_RATIO 影响')
    axes[1][1].axis('off')
    axes[1][2].axis('off')

    # ── 十字星：DOJI_THRESHOLD 对比 ───────────────────────────────────────
    ax = axes[0][3]
    # 画三根K线：实体分别为 0.5×, 1.0×, 2.0× 阈值
    for i, mult in enumerate([-0.25, 0, 0.25]):
        body_pct = C.DOJI_THRESHOLD * [0.5, 1.0, 2.0][i]
        body = price * body_pct
        o = price - body/2
        c = price + body/2
        h = price + price*0.015
        l = price - price*0.015
        draw_candle(ax, mult, o, h, l, c, width=0.18)
        is_valid = body_pct <= C.DOJI_THRESHOLD
        ax.text(mult, l - price*0.003,
                f'{body_pct:.4f}p\n{"✓" if is_valid else "✗"}',
                ha='center', fontsize=7,
                color='#26a69a' if is_valid else '#ef5350')
    setup_candle_ax(ax, f'十字星 DOJI_THRESHOLD={C.DOJI_THRESHOLD}',
                    ylim=(price*0.982, price*1.018))
    ax.text(0, price*1.016, f'阈值线 = {C.DOJI_THRESHOLD}p',
            ha='center', fontsize=7.5, color='#DD4444')

    param_box(axes[1][3], [
        (f'当前 DOJI_THRESHOLD = {C.DOJI_THRESHOLD}', '#DD4444'),
        (f'左({C.DOJI_THRESHOLD*0.5:.4f}p): 极细实体 → 触发', '#26a69a'),
        (f'中({C.DOJI_THRESHOLD:.4f}p): 边界 → 触发', '#26a69a'),
        (f'右({C.DOJI_THRESHOLD*2:.4f}p): 超出阈值 → 不触发', '#ef5350'),
    ], 'DOJI_THRESHOLD 影响')

    fig.tight_layout(h_pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('生成单根K线形态图...')
    f1 = fig_single_candles()
    f1.savefig('candlestick_single.png', dpi=130, bbox_inches='tight')

    print('生成双根K线形态图...')
    f2 = fig_double_candles()
    f2.savefig('candlestick_double.png', dpi=130, bbox_inches='tight')

    print('生成三根K线形态图...')
    f3 = fig_triple_candles()
    f3.savefig('candlestick_triple.png', dpi=130, bbox_inches='tight')

    print('生成价格位置阈值图...')
    f4 = fig_price_pos()
    f4.savefig('candlestick_price_pos.png', dpi=130, bbox_inches='tight')

    print('生成参数影响对比图...')
    f5 = fig_param_impact()
    f5.savefig('candlestick_param_impact.png', dpi=130, bbox_inches='tight')

    plt.close('all')
    print('\n已保存：')
    print('  candlestick_single.png      — 单根形态（十字星/锤子/射击之星等）')
    print('  candlestick_double.png      — 双根形态（吞没/刺穿/乌云/孕线）')
    print('  candlestick_triple.png      — 三根形态（晨星/暮星/三白兵/三乌鸦）')
    print('  candlestick_price_pos.png   — 价格位置阈值示意')
    print('  candlestick_param_impact.png — 参数调整影响对比')
