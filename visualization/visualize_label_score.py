"""
标签评分可视化脚本

功能：
1. 模拟典型走势，对比各项得分分解
2. 随机读取1000只股票某日截面数据，绘制得分分布图
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sqlite3
import talib
import warnings
warnings.filterwarnings('ignore')

from config.factor_config import TrainingConfig
from config import strategy_config, DATABASE_PATH

# ── 字体设置（支持中文）──────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

EPS = 1e-4
FORWARD_DAYS = TrainingConfig.FUTURE_DAYS
ATR_PERIOD    = strategy_config.ATR_PERIOD


# ══════════════════════════════════════════════════════════════════════════════
# 核心评分函数（与 train_ml_model.py 保持一致）
# ══════════════════════════════════════════════════════════════════════════════

def compute_score(f_returns, f_low_min, f_high_idx, f_low_idx, atr_raw, next_open, rel_atr):
    eps = EPS
    core_term    = f_returns / (rel_atr ** 0.5) * TrainingConfig.LABEL_TARGET_SCALE
    lambda_val   = TrainingConfig.LABEL_LAMBDA
    downside_gap = np.maximum(next_open - f_low_min, 0) / (atr_raw + eps)

    linear_part    = np.minimum(downside_gap, 1.0)
    nonlinear_part = np.maximum(downside_gap - 1.0, 0.0) ** 2
    loss_aversion  = -lambda_val * (linear_part + nonlinear_part)

    path_punish_coef = TrainingConfig.LABEL_PATH_PUNISH
    is_v_shape   = (f_low_idx < f_high_idx) & (downside_gap > 1.0)
    path_penalty = np.where(
        is_v_shape & (core_term > 0),
        -path_punish_coef * np.clip(downside_gap - 1.0, 0, 1.5),
        0
    )

    time_bonus = np.where(
        (f_high_idx < 3) & (core_term > 0.5),
        TrainingConfig.LABEL_TIME_BONUS * core_term,
        0
    )

    return (core_term + loss_aversion + path_penalty + time_bonus,
            core_term, loss_aversion, path_penalty, time_bonus)


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — 典型走势模拟
# ══════════════════════════════════════════════════════════════════════════════

def build_scenario(name, prices_7d, base_price=10.0, atr_pct=0.02):
    """
    prices_7d: 长度为 FORWARD_DAYS 的相对涨跌序列（相对 base_price）
    返回单行标量得分及分解
    """
    prices = np.array([base_price] + [base_price * (1 + r) for r in np.cumsum(prices_7d)])
    close_arr = prices[:-1]   # T 日收盘
    high_arr  = close_arr * 1.005
    low_arr   = close_arr * 0.995
    next_open = base_price    # T+1 开盘 = 买入价

    # 未来 FORWARD_DAYS 日
    fut_prices = prices[1:]
    f_close    = fut_prices[-1]
    f_high_max = fut_prices.max()
    f_low_min  = fut_prices.min()
    f_high_idx = float(np.argmax(fut_prices))
    f_low_idx  = float(np.argmin(fut_prices))

    f_returns  = f_close / next_open - 1
    atr_raw    = base_price * atr_pct
    rel_atr    = atr_pct

    score, core, loss_av, path_pen, t_bonus = compute_score(
        np.array([f_returns]), np.array([f_low_min]),
        np.array([f_high_idx]), np.array([f_low_idx]),
        np.array([atr_raw]), np.array([next_open]), np.array([rel_atr])
    )
    return {
        'name': name,
        'f_returns_pct': f_returns * 100,
        'f_low_min_pct': (f_low_min / next_open - 1) * 100,
        'f_high_idx': f_high_idx,
        'f_low_idx': f_low_idx,
        'score': score[0],
        'core': core[0],
        'loss_aversion': loss_av[0],
        'path_penalty': path_pen[0],
        'time_bonus': t_bonus[0],
        'prices': prices,
    }


SCENARIOS = [
    build_scenario('稳步上涨',    [0.01, 0.015, 0.02, 0.015, 0.01, 0.02, 0.025]),
    build_scenario('快速拉升\n(高点在第1天)', [0.05, 0.01, -0.01, 0.0, 0.01, 0.0, 0.02]),
    build_scenario('V型反转\n(先跌后涨)', [-0.03, -0.02, 0.01, 0.03, 0.02, 0.02, 0.03]),
    build_scenario('高位震荡\n(涨后横盘)', [0.04, 0.0, -0.01, 0.0, 0.01, -0.01, 0.0]),
    build_scenario('冲高回落\n(最终亏损)', [0.04, 0.02, -0.01, -0.02, -0.02, -0.01, -0.02]),
    build_scenario('持续下跌',    [-0.01, -0.02, -0.015, -0.01, -0.02, -0.015, -0.01]),
    build_scenario('低波动微涨',  [0.003, 0.004, 0.003, 0.005, 0.004, 0.003, 0.004],  atr_pct=0.008),
    build_scenario('高波动大涨',  [0.02, 0.03, -0.01, 0.02, 0.03, 0.01, 0.02],        atr_pct=0.04),
]


def plot_scenarios(scenarios, save_path):
    n = len(scenarios)
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    # ── 子图1：价格走势 ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, s in enumerate(scenarios):
        ax1.plot(s['prices'], marker='o', markersize=3, color=colors[i],
                 label=f"{s['name'].replace(chr(10),' ')}  ({s['f_returns_pct']:+.1f}%)")
    ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='买入点(T+1开盘)')
    ax1.set_title('典型走势模拟（基准价=10元，持有7日）', fontsize=13)
    ax1.set_xlabel('日期偏移')
    ax1.set_ylabel('价格')
    ax1.legend(fontsize=7.5, ncol=2, loc='upper left')
    ax1.grid(alpha=0.3)

    # ── 子图2：总得分对比 ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    names  = [s['name'] for s in scenarios]
    scores = [s['score'] for s in scenarios]
    bar_colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in scores]
    bars = ax2.bar(range(n), scores, color=bar_colors, edgecolor='white', linewidth=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, val + (0.03 if val >= 0 else -0.08),
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_title('最终得分对比', fontsize=13)
    ax2.set_ylabel('path_quality_score')
    ax2.grid(axis='y', alpha=0.3)

    # ── 子图3：得分分解堆叠图 ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    components = {
        'core_term':     [s['core'] for s in scenarios],
        'loss_aversion': [s['loss_aversion'] for s in scenarios],
        'path_penalty':  [s['path_penalty'] for s in scenarios],
        'time_bonus':    [s['time_bonus'] for s in scenarios],
    }
    comp_colors = {'core_term': '#3498db', 'loss_aversion': '#e74c3c',
                   'path_penalty': '#e67e22', 'time_bonus': '#2ecc71'}
    bottoms_pos = np.zeros(n)
    bottoms_neg = np.zeros(n)
    for comp_name, vals in components.items():
        vals = np.array(vals)
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        ax3.bar(range(n), pos_vals, bottom=bottoms_pos,
                color=comp_colors[comp_name], label=comp_name, edgecolor='white', linewidth=0.5)
        ax3.bar(range(n), neg_vals, bottom=bottoms_neg,
                color=comp_colors[comp_name], edgecolor='white', linewidth=0.5)
        bottoms_pos += pos_vals
        bottoms_neg += neg_vals

    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(names, fontsize=9)
    ax3.set_title('得分分解（各项贡献）', fontsize=13)
    ax3.set_ylabel('得分贡献')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(axis='y', alpha=0.3)

    plt.suptitle('标签评分函数分析 — 典型走势', fontsize=15, fontweight='bold', y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Part1] 已保存: {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — 真实截面数据得分分布
# ══════════════════════════════════════════════════════════════════════════════

def load_cross_section(n_stocks=1000):
    """
    随机抽取 n_stocks 只股票，找到一个共同有效交易日，
    计算每只股票在该日的 path_quality_score。
    返回 DataFrame，每行一只股票。
    """
    conn = sqlite3.connect(DATABASE_PATH)

    # 找最近有足够数据的日期（需要 ATR_PERIOD + FORWARD_DAYS 天历史）
    all_codes = pd.read_sql(
        "SELECT DISTINCT code FROM daily_data ORDER BY RANDOM() LIMIT ?",
        conn, params=(n_stocks,)
    )['code'].tolist()

    # 取最新公共日期（最近30个交易日内）
    recent_dates = pd.read_sql(
        "SELECT DISTINCT date FROM daily_data ORDER BY date DESC LIMIT 60",
        conn
    )['date'].tolist()

    # 选一个往前偏移 FORWARD_DAYS+2 的日期，确保未来数据存在
    offset = FORWARD_DAYS + 2
    target_date = None
    for d in recent_dates[offset: offset + 15]:
        cnt = pd.read_sql(
            "SELECT COUNT(DISTINCT code) as cnt FROM daily_data WHERE date=?",
            conn, params=(d,)
        )['cnt'].iloc[0]
        if cnt >= 500:
            target_date = d
            break

    print(f'[Part2] 截面日期: {target_date}，覆盖股票数: {cnt}')

    results = []
    min_history = ATR_PERIOD + FORWARD_DAYS + 5

    for code in all_codes:
        try:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, is_st FROM daily_data "
                "WHERE code=? AND date<=? ORDER BY date",
                conn, params=(code, target_date)
            )
            if len(df) < min_history:
                continue

            close = df['close'].values
            high  = df['high'].values
            low   = df['low'].values
            opens = df['open'].values

            # 加载截面日之后的未来数据
            df_fut = pd.read_sql(
                "SELECT date, open, high, low, close FROM daily_data "
                "WHERE code=? AND date>? ORDER BY date LIMIT ?",
                conn, params=(code, target_date, FORWARD_DAYS)
            )
            if len(df_fut) < FORWARD_DAYS:
                continue
            fut_high      = df_fut['high'].values
            fut_low       = df_fut['low'].values
            fut_close     = df_fut['close'].values
            next_open_val = df_fut['open'].iloc[0]

            f_returns  = fut_close[-1] / next_open_val - 1
            f_low_min  = fut_low.min()
            f_high_max = fut_high.max()
            f_high_idx = float(np.argmax(fut_high))
            f_low_idx  = float(np.argmin(fut_low))

            atr_raw = talib.ATR(high, low, close, timeperiod=ATR_PERIOD)[-1]
            if np.isnan(atr_raw) or atr_raw <= 0:
                continue
            rel_atr = atr_raw / close[-1]

            score, core, loss_av, path_pen, t_bonus = compute_score(
                np.array([f_returns]), np.array([f_low_min]),
                np.array([f_high_idx]), np.array([f_low_idx]),
                np.array([atr_raw]), np.array([next_open_val]), np.array([rel_atr])
            )

            results.append({
                'code': code,
                'is_st': int(df['is_st'].iloc[-1]),
                'f_returns_pct': f_returns * 100,
                'f_low_min_pct': (f_low_min / next_open_val - 1) * 100,
                'rel_atr_pct': rel_atr * 100,
                'score': score[0],
                'core': core[0],
                'loss_aversion': loss_av[0],
                'path_penalty': path_pen[0],
                'time_bonus': t_bonus[0],
            })
        except Exception:
            continue

    conn.close()
    df_res = pd.DataFrame(results).dropna()
    print(f'[Part2] 有效样本: {len(df_res)} 只')
    return df_res, target_date


def plot_cross_section(df, target_date, save_path):
    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── 1. 得分总体分布 ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    scores = df['score'].clip(-5, 5)
    ax1.hist(scores, bins=80, color='#3498db', edgecolor='white', linewidth=0.4, alpha=0.85)
    ax1.axvline(scores.mean(), color='red',    linestyle='--', linewidth=1.5, label=f'均值 {scores.mean():.2f}')
    ax1.axvline(scores.median(), color='orange', linestyle='--', linewidth=1.5, label=f'中位数 {scores.median():.2f}')
    ax1.axvline(0, color='black', linewidth=1.0, label='0')
    pct_pos = (df['score'] > 0).mean() * 100
    ax1.set_title(f'截面得分分布（{target_date}，n={len(df)}，正分占比 {pct_pos:.1f}%）', fontsize=13)
    ax1.set_xlabel('path_quality_score（已截断至[-5,5]）')
    ax1.set_ylabel('股票数量')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # ── 2. 各分项分布 ────────────────────────────────────────────────────────
    comp_info = [
        ('core', '#3498db', 'core_term'),
        ('loss_aversion', '#e74c3c', 'loss_aversion'),
        ('path_penalty', '#e67e22', 'path_penalty'),
        ('time_bonus', '#2ecc71', 'time_bonus'),
    ]
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    for (col, color, label), (r, c) in zip(comp_info, positions):
        ax = fig.add_subplot(gs[r, c])
        vals = df[col].clip(-5, 5)
        ax.hist(vals, bins=60, color=color, edgecolor='white', linewidth=0.3, alpha=0.85)
        ax.axvline(vals.mean(), color='black', linestyle='--', linewidth=1.2,
                   label=f'均值 {vals.mean():.3f}')
        nonzero_pct = (vals != 0).mean() * 100
        ax.set_title(f'{label}  (非零占比 {nonzero_pct:.1f}%)', fontsize=11)
        ax.set_xlabel('值')
        ax.set_ylabel('数量')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f'标签评分函数分析 — 真实截面数据 ({target_date})', fontsize=15, fontweight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Part2] 已保存: {save_path}')


def plot_cross_section_scatter(df, target_date, save_path):
    """散点图：得分 vs 收益率 / ATR，以及分位数分析"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'得分与关键变量关系 ({target_date})', fontsize=14, fontweight='bold')

    # 散点1：得分 vs 未来收益率
    ax = axes[0]
    sc = ax.scatter(df['f_returns_pct'], df['score'].clip(-5, 5),
                    c=df['rel_atr_pct'], cmap='RdYlGn_r', alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label='rel_atr (%)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('未来7日收益率 (%)')
    ax.set_ylabel('path_quality_score')
    ax.set_title('得分 vs 未来收益率\n（颜色=相对ATR）')
    ax.grid(alpha=0.3)

    # 散点2：得分 vs 相对ATR
    ax = axes[1]
    ax.scatter(df['rel_atr_pct'], df['score'].clip(-5, 5),
               c=df['f_returns_pct'], cmap='RdYlGn', alpha=0.4, s=8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('相对ATR (%)')
    ax.set_ylabel('path_quality_score')
    ax.set_title('得分 vs 波动率\n（颜色=未来收益率）')
    ax.grid(alpha=0.3)

    # 分位数分析：按得分分10组，看各组平均收益率
    ax = axes[2]
    df2 = df.copy()
    df2['score_decile'] = pd.qcut(df2['score'], q=10, labels=False, duplicates='drop')
    grp = df2.groupby('score_decile')['f_returns_pct'].mean()
    bar_colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in grp.values]
    ax.bar(grp.index, grp.values, color=bar_colors, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('得分分位组（0=最低，9=最高）')
    ax.set_ylabel('组内平均未来收益率 (%)')
    ax.set_title('得分分位 vs 平均未来收益率\n（标签有效性验证）')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Part2] 散点图已保存: {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    out_dir = 'backtest_result/label_analysis'
    os.makedirs(out_dir, exist_ok=True)

    # Part 1
    print('=== Part 1: 典型走势模拟 ===')
    plot_scenarios(SCENARIOS, os.path.join(out_dir, 'label_scenarios.png'))

    # 打印数值表
    print(f"\n{'走势':<20} {'收益%':>7} {'最低%':>7} {'高点日':>6} {'低点日':>6} "
          f"{'core':>7} {'loss_av':>8} {'path_pen':>9} {'t_bonus':>8} {'总分':>7}")
    print('-' * 90)
    for s in SCENARIOS:
        print(f"{s['name'].replace(chr(10),' '):<20} {s['f_returns_pct']:>7.2f} "
              f"{s['f_low_min_pct']:>7.2f} {s['f_high_idx']:>6.0f} {s['f_low_idx']:>6.0f} "
              f"{s['core']:>7.3f} {s['loss_aversion']:>8.3f} {s['path_penalty']:>9.3f} "
              f"{s['time_bonus']:>8.3f} {s['score']:>7.3f}")

    # Part 2
    print('\n=== Part 2: 真实截面数据 ===')
    df_cs, target_date = load_cross_section(n_stocks=1000)
    if len(df_cs) > 50:
        plot_cross_section(df_cs, target_date,
                           os.path.join(out_dir, 'label_distribution.png'))
        plot_cross_section_scatter(df_cs, target_date,
                                   os.path.join(out_dir, 'label_scatter.png'))

        print(f"\n截面得分统计:")
        print(df_cs['score'].describe().round(3).to_string())
        print(f"\n各分项均值:")
        for col in ['core', 'loss_aversion', 'path_penalty', 'time_bonus']:
            nonzero = (df_cs[col] != 0).mean() * 100
            print(f"  {col:<16}: mean={df_cs[col].mean():>7.3f}  std={df_cs[col].std():>6.3f}  非零={nonzero:.1f}%")

        # 额外诊断：中间变量分布
        eps = EPS
        print(f"\n关键中间变量分布（用于参数校准）:")
        print(f"  rel_atr_pct  : mean={df_cs['rel_atr_pct'].mean():.3f}%  "
              f"p25={df_cs['rel_atr_pct'].quantile(0.25):.3f}%  "
              f"p50={df_cs['rel_atr_pct'].median():.3f}%  "
              f"p75={df_cs['rel_atr_pct'].quantile(0.75):.3f}%")
        df_cs['downside_gap'] = (-df_cs['f_low_min_pct'] / 100) / (df_cs['rel_atr_pct'] / 100 + eps)
        print(f"  downside_gap : mean={df_cs['downside_gap'].mean():.3f}  "
              f"p25={df_cs['downside_gap'].quantile(0.25):.3f}  "
              f"p50={df_cs['downside_gap'].median():.3f}  "
              f"p75={df_cs['downside_gap'].quantile(0.75):.3f}  "
              f"p90={df_cs['downside_gap'].quantile(0.90):.3f}")
        print(f"  core_term    : p10={df_cs['core'].quantile(0.10):.3f}  "
              f"p25={df_cs['core'].quantile(0.25):.3f}  "
              f"p50={df_cs['core'].median():.3f}  "
              f"p75={df_cs['core'].quantile(0.75):.3f}  "
              f"p90={df_cs['core'].quantile(0.90):.3f}")
    else:
        print('[警告] 有效样本不足，跳过绘图')

    print(f'\n完成！输出目录: {out_dir}')
