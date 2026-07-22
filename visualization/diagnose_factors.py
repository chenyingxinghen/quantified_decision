"""
因子有效性 & 标签分布诊断脚本

检查内容：
1. 标签分布（离散档位 + 原始分数）
2. 因子 IC / Rank-IC（信息系数）
3. 因子缺失率 & 零方差检测
4. 因子相关性热图（Top 高相关对）
5. 标签与收益率的一致性验证

用法：
    python diagnose_factors.py [--sample N] [--days D] [--output DIR]
"""

import sys, os, warnings, argparse
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import spearmanr, rankdata
from datetime import datetime, timedelta
from tqdm import tqdm

from config.data_config import DATABASE_PATH, MARKET_LIMITS, MARKET_PREFIXES
from config.factor_config import TrainingConfig, FactorConfig, ModelConfig
from core.factors.train_ml_model import MLModelTrainer

# ── 中文字体（Windows）──────────────────────────────────────────────────────
def _setup_chinese_font():
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong']
    for name in candidates:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False
            return name
    return None

FONT_NAME = _setup_chinese_font()


# ============================================================================
# 1. 数据加载
# ============================================================================

def load_sample_stocks(n_stocks: int = 200, years: float = 3.0) -> dict:
    """从数据库随机抽取 n_stocks 只股票的行情数据"""
    end_date = (datetime.today() - timedelta(days=int(years * 365))).strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=int(years*2 * 365))).strftime('%Y-%m-%d')

    conn = sqlite3.connect(DATABASE_PATH)
    db_dir = os.path.dirname(DATABASE_PATH)
    meta_db = os.path.join(db_dir, 'stock_meta.db')
    if os.path.exists(meta_db):
        conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")

    # 取满足策略过滤条件的股票池
    markets = TrainingConfig.TRAIN_FILTER_MARKETS or ['sh_main', 'sz_main']
    prefixes = []
    for m in markets:
        if m == 'sh_main':   prefixes.append('60')
        elif m == 'sz_main': prefixes.append('00')
        elif m == 'sz_gem':  prefixes.append('30')
        elif m == 'sh_star': prefixes.append('68')

    like_clauses = ' OR '.join([f"code LIKE '{p}%'" for p in prefixes]) if prefixes else '1=1'

    try:
        all_codes_df = pd.read_sql_query(
            f"SELECT DISTINCT code FROM daily_data WHERE date >= ? AND ({like_clauses})",
            conn, params=[start_date]
        )
        all_codes = all_codes_df['code'].tolist()
    except Exception as e:
        print(f"  ⚠ 读取股票列表失败: {e}")
        conn.close()
        return {}

    import random
    random.seed(42)
    sample_codes = random.sample(all_codes, min(n_stocks, len(all_codes)))
    print(f"  抽取 {len(sample_codes)} 只股票 ({start_date} ~ {end_date})")

    placeholders = ','.join(['?'] * len(sample_codes))
    query = f"""
        SELECT k.code, k.date, k.open, k.high, k.low, k.close, k.preclose,
               k.volume, k.amount, k.turnover_rate, k.is_st, k.peTTM, k.pbMRQ,
               a.fore_adjust_factor
        FROM daily_data k
        LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
        WHERE k.code IN ({placeholders}) AND k.date >= ? AND k.date <= ?
        ORDER BY k.code, k.date
    """
    df = pd.read_sql_query(query, conn, params=sample_codes + [start_date, end_date])
    conn.close()

    stocks_data = {}
    for code, grp in df.groupby('code'):
        grp = grp.sort_values('date').reset_index(drop=True)
        if len(grp) < 120:
            continue
        # 简单前复权
        if 'fore_adjust_factor' in grp.columns:
            valid_adj = grp['fore_adjust_factor'].dropna()
            if not valid_adj.empty:
                base = float(valid_adj.iloc[-1])
                if base != 0:
                    ratio = grp['fore_adjust_factor'].bfill().ffill().fillna(1.0) / base
                    for col in ['open', 'high', 'low', 'close', 'preclose']:
                        if col in grp.columns:
                            grp[col] = grp[col] * ratio
        grp['days_to_delist'] = np.float32(-1)
        stocks_data[code] = grp

    print(f"  有效股票: {len(stocks_data)} 只")
    return stocks_data


# ============================================================================
# 2. 标签分布分析
# ============================================================================

def analyze_label_distribution(raw_scores: np.ndarray, y_ranked: np.ndarray,
                                y_discrete: np.ndarray, returns: np.ndarray,
                                output_dir: str):
    """绘制标签分布图并打印统计"""
    os.makedirs(output_dir, exist_ok=True)
    n_bins = ModelConfig.get_n_bins()

    print("\n" + "="*60)
    print("【标签分布分析】")
    print("="*60)

    # ── 离散档位分布 ──────────────────────────────────────────────────────
    unique, counts = np.unique(y_discrete, return_counts=True)
    total = len(y_discrete)
    print(f"\n离散档位分布 (N_BINS={n_bins}):")
    print(f"  {'档位':>4}  {'样本数':>8}  {'占比':>7}  {'累计':>7}")
    cumsum = 0
    for b, c in zip(unique, counts):
        cumsum += c
        print(f"  {b:>4}  {c:>8,}  {c/total*100:>6.2f}%  {cumsum/total*100:>6.2f}%")

    # ── 原始分数统计 ──────────────────────────────────────────────────────
    print(f"\n原始路径质量分数统计:")
    print(f"  均值={raw_scores.mean():.4f}  中位数={np.median(raw_scores):.4f}")
    print(f"  标准差={raw_scores.std():.4f}  偏度={pd.Series(raw_scores).skew():.4f}")
    print(f"  P5={np.percentile(raw_scores,5):.4f}  P25={np.percentile(raw_scores,25):.4f}")
    print(f"  P75={np.percentile(raw_scores,75):.4f}  P95={np.percentile(raw_scores,95):.4f}")

    # ── 收益率统计 ────────────────────────────────────────────────────────
    print(f"\n未来{TrainingConfig.FUTURE_DAYS}日收益率统计:")
    print(f"  均值={returns.mean()*100:.2f}%  中位数={np.median(returns)*100:.2f}%")
    print(f"  标准差={returns.std()*100:.2f}%")
    print(f"  正收益占比: {(returns > 0).mean()*100:.1f}%")
    print(f"  负收益占比: {(returns < 0).mean()*100:.1f}%")

    # ── 各档位平均收益率（验证标签单调性）────────────────────────────────
    print(f"\n各档位平均收益率（单调性验证）:")
    bin_returns = {}
    for b in sorted(unique):
        mask = y_discrete == b
        avg_ret = returns[mask].mean() * 100
        bin_returns[b] = avg_ret
        bar = '█' * int(abs(avg_ret) * 5) if abs(avg_ret) < 20 else '█' * 20
        sign = '+' if avg_ret >= 0 else ''
        print(f"  档位{b:>2}: {sign}{avg_ret:>6.2f}%  {bar}")

    # ── 绘图 ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'标签分布诊断  (FUTURE_DAYS={TrainingConfig.FUTURE_DAYS}, N_BINS={n_bins})',
                 fontsize=14, fontweight='bold')

    # 子图1：离散档位分布
    ax = axes[0, 0]
    ax.bar(unique, counts / total * 100, color='steelblue', edgecolor='white')
    ax.set_xlabel('档位')
    ax.set_ylabel('占比 (%)')
    ax.set_title('离散标签档位分布')
    ax.set_xticks(range(n_bins))

    # 子图2：原始分数直方图
    ax = axes[0, 1]
    ax.hist(raw_scores, bins=100, color='coral', edgecolor='none', alpha=0.8)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('路径质量原始分')
    ax.set_ylabel('频次')
    ax.set_title('原始分数分布')

    # 子图3：各档位平均收益率
    ax = axes[1, 0]
    bins_sorted = sorted(bin_returns.keys())
    rets = [bin_returns[b] for b in bins_sorted]
    colors = ['green' if r >= 0 else 'red' for r in rets]
    ax.bar(bins_sorted, rets, color=colors, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('档位')
    ax.set_ylabel('平均收益率 (%)')
    ax.set_title('各档位平均收益率（单调性）')
    ax.set_xticks(range(n_bins))

    # 子图4：收益率分布
    ax = axes[1, 1]
    ax.hist(returns * 100, bins=100, color='mediumpurple', edgecolor='none', alpha=0.8)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(f'未来{TrainingConfig.FUTURE_DAYS}日收益率 (%)')
    ax.set_ylabel('频次')
    ax.set_title('收益率分布')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'label_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ 标签分布图已保存: {save_path}")

    # ── 单调性评分 ────────────────────────────────────────────────────────
    rets_arr = np.array(rets)
    monotone_pairs = sum(1 for i in range(len(rets_arr)-1) if rets_arr[i] <= rets_arr[i+1])
    total_pairs = len(rets_arr) - 1
    monotone_score = monotone_pairs / total_pairs * 100
    print(f"\n  标签单调性得分: {monotone_score:.1f}% ({monotone_pairs}/{total_pairs} 对单调递增)")
    if monotone_score >= 80:
        print("  ✓ 标签单调性良好，档位与收益率正相关")
    elif monotone_score >= 60:
        print("  ⚠ 标签单调性一般，部分档位收益率不单调，建议检查标签构造逻辑")
    else:
        print("  ✗ 标签单调性差，档位与收益率相关性弱，模型可能无法有效学习")


# ============================================================================
# 3. 因子有效性分析（IC / Rank-IC）
# ============================================================================

def compute_factor_ic(X: np.ndarray, returns: np.ndarray, dates: np.ndarray,
                      factor_names: list, top_n: int = 30) -> pd.DataFrame:
    """
    计算每个因子的截面 IC（Pearson）和 Rank-IC（Spearman）。
    返回按 |mean_rank_ic| 降序排列的 DataFrame。
    """
    print("\n" + "="*60)
    print("【因子 IC / Rank-IC 分析】")
    print("="*60)

    unique_dates = np.unique(dates)
    n_factors = X.shape[1]

    ic_matrix = np.full((len(unique_dates), n_factors), np.nan)
    ric_matrix = np.full((len(unique_dates), n_factors), np.nan)

    for di, d in enumerate(tqdm(unique_dates, desc="计算截面IC", ncols=80)):
        mask = dates == d
        if mask.sum() < 10:
            continue
        X_d = X[mask]
        r_d = returns[mask]
        if np.isnan(r_d).any():
            continue
        for fi in range(n_factors):
            col = X_d[:, fi]
            if np.isnan(col).any() or col.std() < 1e-8:
                continue
            try:
                ic_matrix[di, fi] = np.corrcoef(col, r_d)[0, 1]
                ric_matrix[di, fi], _ = spearmanr(col, r_d)
            except Exception:
                pass

    mean_ic  = np.nanmean(ic_matrix,  axis=0)
    mean_ric = np.nanmean(ric_matrix, axis=0)
    std_ric  = np.nanstd(ric_matrix,  axis=0)
    icir     = np.where(std_ric > 1e-8, mean_ric / std_ric, 0.0)
    valid_days = np.sum(~np.isnan(ric_matrix), axis=0)

    ic_df = pd.DataFrame({
        'factor': factor_names,
        'mean_ic':   mean_ic,
        'mean_ric':  mean_ric,
        'std_ric':   std_ric,
        'icir':      icir,
        'valid_days': valid_days,
        'abs_ric':   np.abs(mean_ric)
    }).sort_values('abs_ric', ascending=False).reset_index(drop=True)

    # ── 打印 Top-N ────────────────────────────────────────────────────────
    print(f"\nTop-{top_n} 因子（按 |Rank-IC| 排序）:")
    print(f"  {'因子名':<35} {'IC':>7} {'RankIC':>8} {'ICIR':>7} {'有效天':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")
    for _, row in ic_df.head(top_n).iterrows():
        flag = '★' if abs(row['mean_ric']) >= 0.03 else ' '
        print(f"  {flag}{row['factor']:<34} {row['mean_ic']:>7.4f} {row['mean_ric']:>8.4f} "
              f"{row['icir']:>7.3f} {int(row['valid_days']):>6}")

    # ── 汇总统计 ──────────────────────────────────────────────────────────
    valid_factors = ic_df[ic_df['valid_days'] > 0]
    strong = (valid_factors['abs_ric'] >= 0.03).sum()
    moderate = ((valid_factors['abs_ric'] >= 0.01) & (valid_factors['abs_ric'] < 0.03)).sum()
    weak = (valid_factors['abs_ric'] < 0.01).sum()

    print(f"\n因子有效性汇总 (共 {len(valid_factors)} 个有效因子):")
    print(f"  强有效 (|RankIC|≥0.03): {strong:>4} 个  ({strong/len(valid_factors)*100:.1f}%)")
    print(f"  中等   (0.01~0.03):     {moderate:>4} 个  ({moderate/len(valid_factors)*100:.1f}%)")
    print(f"  弱/无效 (<0.01):        {weak:>4} 个  ({weak/len(valid_factors)*100:.1f}%)")
    print(f"  平均 |RankIC|: {valid_factors['abs_ric'].mean():.4f}")
    print(f"  平均 ICIR:     {valid_factors['icir'].abs().mean():.3f}")

    return ic_df, ic_matrix, ric_matrix, unique_dates


# ============================================================================
# 4. 因子质量检查（缺失率、零方差、相关性）
# ============================================================================

def analyze_factor_quality(X: np.ndarray, factor_names: list, output_dir: str):
    """检查因子缺失率、零方差、高相关对"""
    print("\n" + "="*60)
    print("【因子质量检查】")
    print("="*60)

    n_samples, n_factors = X.shape

    # ── 缺失率 ────────────────────────────────────────────────────────────
    nan_rates = np.isnan(X).mean(axis=0)
    high_nan = [(factor_names[i], nan_rates[i]) for i in range(n_factors) if nan_rates[i] > 0.1]
    high_nan.sort(key=lambda x: -x[1])

    print(f"\n缺失率 > 10% 的因子 ({len(high_nan)} 个):")
    if high_nan:
        for name, rate in high_nan[:20]:
            print(f"  {name:<40} {rate*100:.1f}%")
        if len(high_nan) > 20:
            print(f"  ... 还有 {len(high_nan)-20} 个")
    else:
        print("  ✓ 无高缺失率因子")

    # ── 零方差 ────────────────────────────────────────────────────────────
    stds = np.nanstd(X, axis=0)
    zero_var = [(factor_names[i], stds[i]) for i in range(n_factors) if stds[i] < 1e-6]
    print(f"\n零方差因子 ({len(zero_var)} 个):")
    if zero_var:
        for name, std in zero_var[:20]:
            print(f"  {name:<40} std={std:.2e}")
    else:
        print("  ✓ 无零方差因子")

    # ── 高相关对（抽样计算，避免内存爆炸）────────────────────────────────
    print(f"\n高相关因子对检测 (|corr| > 0.95):")
    # 只取前 100 个因子做相关性矩阵（避免 OOM）
    max_corr_factors = min(100, n_factors)
    X_sub = X[:, :max_corr_factors]
    # 用 nanmean 填充 NaN 再计算
    col_means = np.nanmean(X_sub, axis=0)
    for i in range(X_sub.shape[1]):
        nan_mask = np.isnan(X_sub[:, i])
        X_sub[nan_mask, i] = col_means[i]

    corr_mat = np.corrcoef(X_sub.T)
    high_corr_pairs = []
    for i in range(max_corr_factors):
        for j in range(i+1, max_corr_factors):
            c = corr_mat[i, j]
            if abs(c) > 0.95:
                high_corr_pairs.append((factor_names[i], factor_names[j], c))

    high_corr_pairs.sort(key=lambda x: -abs(x[2]))
    if high_corr_pairs:
        print(f"  发现 {len(high_corr_pairs)} 对高相关因子（前20对）:")
        for a, b, c in high_corr_pairs[:20]:
            print(f"  {a:<35} ↔ {b:<35}  corr={c:.3f}")
    else:
        print(f"  ✓ 前 {max_corr_factors} 个因子中无高相关对")

    # ── 绘制因子缺失率分布 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('因子质量检查', fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.hist(nan_rates * 100, bins=50, color='steelblue', edgecolor='none')
    ax.axvline(10, color='red', linestyle='--', linewidth=1.5, label='10% 阈值')
    ax.set_xlabel('缺失率 (%)')
    ax.set_ylabel('因子数量')
    ax.set_title('因子缺失率分布')
    ax.legend()

    ax = axes[1]
    ax.hist(stds, bins=50, color='coral', edgecolor='none')
    ax.axvline(1e-6, color='red', linestyle='--', linewidth=1.5, label='零方差阈值')
    ax.set_xlabel('标准差')
    ax.set_ylabel('因子数量')
    ax.set_title('因子标准差分布')
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'factor_quality.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ 因子质量图已保存: {save_path}")

    return {
        'nan_rates': nan_rates,
        'stds': stds,
        'high_nan_factors': high_nan,
        'zero_var_factors': zero_var,
        'high_corr_pairs': high_corr_pairs,
    }


# ============================================================================
# 5. IC 时序图（因子稳定性）
# ============================================================================

def plot_ic_timeseries(ic_df: pd.DataFrame, ric_matrix: np.ndarray,
                       unique_dates: np.ndarray, factor_names: list,
                       output_dir: str, top_n: int = 10):
    """绘制 Top-N 因子的 Rank-IC 时序图"""
    top_factors = ic_df.head(top_n)['factor'].tolist()
    top_indices = [factor_names.index(f) for f in top_factors if f in factor_names]

    if not top_indices:
        return

    fig, axes = plt.subplots(len(top_indices), 1, figsize=(14, 3 * len(top_indices)))
    if len(top_indices) == 1:
        axes = [axes]
    fig.suptitle(f'Top-{top_n} 因子 Rank-IC 时序', fontsize=13, fontweight='bold')

    # 日期转为 pandas DatetimeIndex 用于绘图
    try:
        date_idx = pd.to_datetime(unique_dates)
    except Exception:
        date_idx = np.arange(len(unique_dates))

    for ax, fi, fname in zip(axes, top_indices, top_factors):
        ric_series = ric_matrix[:, fi]
        valid = ~np.isnan(ric_series)
        if valid.sum() == 0:
            continue
        x = date_idx[valid] if hasattr(date_idx, '__getitem__') else np.where(valid)[0]
        y = ric_series[valid]
        # 滚动均值（30日）
        y_roll = pd.Series(y).rolling(30, min_periods=5).mean().values

        ax.bar(x, y, color=np.where(y >= 0, 'steelblue', 'coral'), alpha=0.5, width=1)
        ax.plot(x, y_roll, color='black', linewidth=1.2, label='30日均值')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.axhline(0.03, color='green', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.axhline(-0.03, color='red', linewidth=0.8, linestyle='--', alpha=0.6)
        mean_ric = np.nanmean(ric_series)
        ax.set_title(f'{fname}  (均值RankIC={mean_ric:.4f})', fontsize=9)
        ax.set_ylabel('Rank-IC')
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ic_timeseries.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✓ IC 时序图已保存: {save_path}")


# ============================================================================
# 6. 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='因子有效性 & 标签分布诊断')
    parser.add_argument('--sample', type=int, default=300,
                        help='抽样股票数量 (默认 300)')
    parser.add_argument('--days', type=float, default=3.0,
                        help='历史数据年数 (默认 3.0)')
    parser.add_argument('--output', type=str, default='diagnose_output',
                        help='输出目录 (默认 diagnose_output)')
    parser.add_argument('--top-ic', type=int, default=30,
                        help='展示 Top-N IC 因子 (默认 30)')
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  因子有效性 & 标签分布诊断工具")
    print(f"  FUTURE_DAYS={TrainingConfig.FUTURE_DAYS}  N_BINS={ModelConfig.get_n_bins()}")
    print(f"  SHORT_PREDICTION={TrainingConfig.SHORT_PREDICTION}")
    print("=" * 60)

    # ── Step 1: 加载数据 ──────────────────────────────────────────────────
    print(f"\n[Step 1] 加载行情数据 (抽样 {args.sample} 只, {args.days} 年)...")
    stocks_data = load_sample_stocks(n_stocks=args.sample, years=args.days)
    if not stocks_data:
        print("  ✗ 数据加载失败，退出")
        return

    # ── Step 2: 计算因子 & 构造标签 ──────────────────────────────────────
    print(f"\n[Step 2] 计算因子 & 构造标签...")
    trainer = MLModelTrainer(db_path=DATABASE_PATH)

    end_date = max(d['date'].max() for d in stocks_data.values())
    start_date = min(d['date'].min() for d in stocks_data.values())

    try:
        result = trainer.prepare_dataset(
            stocks_data=stocks_data,
            forward_days=TrainingConfig.FUTURE_DAYS,
            n_jobs=8,
            cache_engineered_features=True,
            train_start_date=start_date,
            train_end_date=end_date,
            include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        )
    except Exception as e:
        import traceback
        print(f"  ✗ prepare_dataset 失败: {e}")
        traceback.print_exc()
        return

    X, y_ranked, returns, factor_names, dates, unbuyable, limit_groups, raw_scores, is_st, w_sig = result
    print(f"\n  数据集: {X.shape[0]:,} 样本 × {X.shape[1]} 因子")

    # ── Step 3: 构造离散标签（复用 train_models 中的逻辑）────────────────
    print(f"\n[Step 3] 构造离散标签...")
    n_bins = ModelConfig.get_n_bins()
    y_discrete = np.empty(len(dates), dtype=np.int32)
    unique_d, d_starts, d_counts = np.unique(dates, return_index=True, return_counts=True)
    for ds, dc in zip(d_starts, d_counts):
        de = ds + dc
        scores = raw_scores[ds:de]
        if dc > 1:
            try:
                
                nbin0_ws = 0.1
                q_skewed = np.concatenate([
                    [0.0, nbin0_ws],
                    np.linspace(nbin0_ws, 1.0, n_bins)
                ])
                bins = pd.qcut(scores, q=q_skewed, labels=False, duplicates='drop')
                y_discrete[ds:de] = bins.astype(np.int32)
            except ValueError:
                ranks = rankdata(scores, method='average') / (dc + 1)
                y_discrete[ds:de] = np.clip((ranks * n_bins).astype(np.int32), 0, n_bins - 1)
        else:
            y_discrete[ds:de] = n_bins // 2

    # ── Step 4: 标签分布分析 ──────────────────────────────────────────────
    print(f"\n[Step 4] 标签分布分析...")
    analyze_label_distribution(raw_scores, y_ranked, y_discrete, returns, output_dir)

    # # ── Step 5: 因子质量检查 ──────────────────────────────────────────────
    # print(f"\n[Step 5] 因子质量检查...")
    # quality_stats = analyze_factor_quality(X, factor_names, output_dir)

    # # ── Step 6: 因子 IC 分析 ──────────────────────────────────────────────
    # print(f"\n[Step 6] 因子 IC / Rank-IC 分析...")
    # # 横截面归一化后再算 IC（与训练时一致）
    # X_norm = X.copy()
    # trainer._apply_cross_sectional_normalization_inplace(X_norm, dates, factor_names)
    # ic_df, ic_matrix, ric_matrix, unique_dates_ic = compute_factor_ic(
    #     X_norm, returns, dates, factor_names, top_n=args.top_ic
    # )

    # # ── Step 7: IC 时序图 ─────────────────────────────────────────────────
    # print(f"\n[Step 7] 绘制 IC 时序图...")
    # plot_ic_timeseries(ic_df, ric_matrix, unique_dates_ic, factor_names, output_dir, top_n=10)

    # # ── Step 8: 保存 IC 报告 ──────────────────────────────────────────────
    # ic_report_path = os.path.join(output_dir, 'factor_ic_report.csv')
    # ic_df.drop(columns=['abs_ric'], errors='ignore').to_csv(ic_report_path, index=False, encoding='utf-8-sig')
    # print(f"  ✓ IC 报告已保存: {ic_report_path}")

    # ── 最终汇总 ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  诊断完成，输出文件:")
    for f in os.listdir(output_dir):
        print(f"    {output_dir}/{f}")
    print("="*60)


if __name__ == '__main__':
    main()
