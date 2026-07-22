"""
因子重要性分析脚本

用法:
    python show_factor_importance.py                        # 使用 models/latest 目录
    python show_factor_importance.py --model models/latest  # 指定模型目录或 .pkl 文件
    python show_factor_importance.py --top 30               # 显示 Top-30（默认 20）
    python show_factor_importance.py --type gain            # 重要性类型: gain/split/cover（XGBoost）
    python show_factor_importance.py --save                 # 同时保存 CSV 和图表
"""

import sys
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='模型因子重要性分析')
    parser.add_argument('--model', type=str, default='models/latest',
                        help='模型目录或 .pkl 文件路径（默认: models/latest）')
    parser.add_argument('--top', type=int, default=20,
                        help='显示 Top-N 因子（默认: 20）')
    parser.add_argument('--type', type=str, default='gain',
                        choices=['gain', 'split', 'cover'],
                        help='XGBoost 重要性类型（默认: gain）')
    parser.add_argument('--save', action='store_true',
                        help='保存结果到 CSV 和 PNG 图表')
    parser.add_argument('--output', type=str, default='factor_importance',
                        help='输出文件名前缀（默认: factor_importance）')
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str):
    """
    加载单个 .pkl 或目录下所有 .pkl，返回 {model_name: model_obj} 字典。
    支持 MLFactorModel 和 EnsembleFactorModel。
    """
    from core.factors.ml_factor_model import MLFactorModel, EnsembleFactorModel

    path = Path(model_path)
    pkl_files = []

    if path.is_file() and path.suffix == '.pkl':
        pkl_files = [path]
    elif path.is_dir():
        pkl_files = sorted(path.glob('*.pkl'))
        if not pkl_files:
            # 递归找一层子目录
            pkl_files = sorted(path.glob('**/*.pkl'))
    
    if not pkl_files:
        raise FileNotFoundError(f"在 {model_path} 下未找到任何 .pkl 文件")

    models = {}
    for pkl in pkl_files:
        name = pkl.stem  # e.g. lightgbm_factor_model
        try:
            # 先尝试 EnsembleFactorModel
            try:
                m = EnsembleFactorModel.load_model(str(pkl))
                models[name] = m
                print(f"  [OK] 加载集成模型: {pkl.name}  ({len(m.models)} 个子模型)")
            except Exception:
                m = MLFactorModel()
                m.load_model(str(pkl))
                models[name] = m
                print(f"  [OK] 加载模型: {pkl.name}  (type={m.model_type}, features={len(m.feature_names)})")
        except Exception as e:
            print(f"  [X] 加载失败: {pkl.name} — {e}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 重要性提取
# ─────────────────────────────────────────────────────────────────────────────
def extract_importance(model, importance_type: str = 'gain') -> pd.Series:
    """
    从单个 MLFactorModel 提取因子重要性，返回按重要性降序排列的 Series。
    importance_type 仅对 XGBoost 原生 Booster 有效（gain/split/cover）。
    """
    import xgboost as xgb
    import lightgbm as lgb

    m = model.model
    feature_names = model.feature_names

    # ── LightGBM ──────────────────────────────────────────────────────────
    if isinstance(m, (lgb.LGBMRanker, lgb.LGBMRegressor, lgb.LGBMClassifier)):
        # LightGBM 支持 gain / split
        imp_type = 'gain' if importance_type in ('gain', 'cover') else 'split'
        raw = m.feature_importances_  # 默认 split，需要用 booster 获取 gain
        if imp_type == 'gain':
            try:
                booster = m.booster_
                raw = booster.feature_importance(importance_type='gain')
            except Exception:
                pass  # 回退到 split
        return pd.Series(raw, index=feature_names).sort_values(ascending=False)

    # ── XGBoost 原生 Booster ───────────────────────────────────────────────
    if isinstance(m, xgb.Booster):
        score = m.get_score(importance_type=importance_type)
        # 映射 f0/f1... → 特征名
        imp = {}
        for k, v in score.items():
            if k.startswith('f') and k[1:].isdigit() and k not in feature_names:
                idx = int(k[1:])
                if idx < len(feature_names):
                    imp[feature_names[idx]] = v
            else:
                imp[k] = v
        s = pd.Series({f: imp.get(f, 0.0) for f in feature_names})
        return s.sort_values(ascending=False)

    # ── XGBoost sklearn 包装 ──────────────────────────────────────────────
    if isinstance(m, (xgb.XGBRegressor, xgb.XGBClassifier)):
        try:
            booster = m.get_booster()
            score = booster.get_score(importance_type=importance_type)
            imp = {}
            for k, v in score.items():
                if k.startswith('f') and k[1:].isdigit() and k not in feature_names:
                    idx = int(k[1:])
                    if idx < len(feature_names):
                        imp[feature_names[idx]] = v
                else:
                    imp[k] = v
            s = pd.Series({f: imp.get(f, 0.0) for f in feature_names})
            return s.sort_values(ascending=False)
        except Exception:
            pass

    # ── 通用回退：feature_importances_ ────────────────────────────────────
    if hasattr(m, 'feature_importances_'):
        return pd.Series(m.feature_importances_, index=feature_names).sort_values(ascending=False)

    raise ValueError(f"无法从模型 {type(m).__name__} 提取重要性")


def extract_ensemble_importance(ensemble_model, importance_type: str = 'gain') -> pd.Series:
    """
    集成模型：对各子模型重要性按权重加权平均，统一到全特征集。
    """
    all_features = set()
    for m in ensemble_model.models:
        all_features.update(m.feature_names)
    all_features = sorted(all_features)

    weighted_sum = pd.Series(0.0, index=all_features)
    for m, w in zip(ensemble_model.models, ensemble_model.weights):
        try:
            imp = extract_importance(m, importance_type)
            # 归一化到 [0,1] 再加权，消除不同模型量纲差异
            total = imp.sum()
            if total > 0:
                imp = imp / total
            # 对齐到全特征集
            imp = imp.reindex(all_features, fill_value=0.0)
            weighted_sum += imp * w
        except Exception as e:
            print(f"  ⚠ 子模型 {m.model_type} 重要性提取失败: {e}")

    return weighted_sum.sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 因子分类（与训练代码保持一致的精确匹配）
# ─────────────────────────────────────────────────────────────────────────────
def _build_factor_sets():
    """构建各类别因子集合（带缓存）"""
    if hasattr(_build_factor_sets, '_cache'):
        return _build_factor_sets._cache

    try:
        from core.factors.quantitative_factors import QuantitativeFactors
        from config import DATABASE_PATH
        _tech = set(QuantitativeFactors(DATABASE_PATH).get_factor_names())
    except Exception:
        # 回退：用已知的技术指标名称前缀/后缀模式匹配
        _tech = set()

    try:
        from core.factors.candlestick_pattern_factors import CandlestickPatternFactors
        _candle = set(CandlestickPatternFactors().get_pattern_names())
    except Exception:
        _candle = set()

    try:
        from core.factors.fundamental_factors import FundamentalFactors
        _fund = set(FundamentalFactors.NUMERIC_COLS) | {
            'dynamic_pe', 'dynamic_pb', 'inv_pe', 'inv_pb', 'market_cap',
            'roe_x_np_growth', 'roe_to_pb', 'peg', 'sue', 'eav'
        }
    except Exception:
        _fund = set()

    try:
        from core.factors.feature_engineering import FeatureEngineer
        _eng = set(FeatureEngineer().get_generated_features())
    except Exception:
        _eng = set()

    _sentiment = {
        'up_ratio', 'strong_up_ratio', 'down_ratio', 'limit_up_ratio',
        'limit_down_ratio', 'mean_return', 'total_volume', 'adv_vol_ratio', 'breadth_ma20'
    }
    _advanced = {
        'hl_range_mean', 'hl_range_std', 'oc_ratio_mean', 'oc_ratio_std',
        'price_volatility_20', 'price_volatility_60', 'price_skewness', 'price_kurtosis',
        'high_position', 'low_position', 'volume_change_rate', 'volume_volatility',
        'price_volume_corr', 'amount_per_volume', 'amount_change_rate',
        'return_5d', 'return_10d', 'return_20d', 'return_60d',
        'momentum_5d', 'momentum_10d', 'momentum_20d', 'acceleration',
        'downside_risk', 'drawdown', 'max_drawdown_20', 'sharpe_ratio',
        'return_skewness', 'return_kurtosis',
    }

    result = dict(tech=_tech, candle=_candle, fund=_fund, eng=_eng,
                  sentiment=_sentiment, advanced=_advanced)
    _build_factor_sets._cache = result
    return result


def classify_factor(name: str) -> str:
    """将因子名映射到类别标签，支持特征工程生成的复合因子名"""
    sets = _build_factor_sets()
    _tech     = sets['tech']
    _candle   = sets['candle']
    _fund     = sets['fund']
    _eng      = sets['eng']
    _sentiment = sets['sentiment']
    _advanced  = sets['advanced']

    # 精确匹配
    if name in _tech:      return '技术指标'
    if name in _candle:    return 'K线形态'
    if name in _fund:      return '基本面'
    if name in _sentiment: return '市场情绪'
    if name in _advanced:  return '高级时序'
    if name in _eng:       return '特征工程'

    # 前缀匹配
    if name.startswith(('industry_', 'sector_')): return '行业'
    if name.startswith(('is_', 'days_to_', 'market_type')): return '状态'

    # 特征工程复合因子：名称中包含操作符关键词（_mul_, _div_, _sub_, _add_, _x_）
    # 且由已知因子名拼接而成
    _eng_ops = ('_mul_', '_div_', '_sub_', '_add_', '_x_', '_log_', 'log_', 'rank_', 'sqrt_')
    if any(op in name for op in _eng_ops):
        return '特征工程'

    # 基本面衍生（含已知基本面列名片段）
    _fund_keywords = ('YOY', 'MBRevenue', 'ROIC', 'EBIT', 'EPS', 'ROE', 'ROA',
                      'Equity', 'Asset', 'Liability', 'Revenue', 'Profit',
                      'currentRatio', 'quickRatio', 'debtToEquity')
    if any(kw in name for kw in _fund_keywords):
        return '基本面'

    # 技术指标模式兜底（当 QuantitativeFactors 初始化失败时）
    _tech_patterns = (
        'rsi_', 'macd', 'kdj_', 'atr_', 'natr_', 'cci_', 'roc_', 'mtm_',
        'adx_', 'aroon_', 'willr_', 'bias_', 'psy_', 'cmo_', 'trix_',
        'bb_', 'ma_', 'ema_', 'vma_', 'volume_ma', 'volume_std',
        'mfi_', 'vr_', 'vroc_', 'vrsi_', 'vmacd', 'adosc',
        'price_var_', 'ulcer_', 'stochrsi_', 'rvi_',
        'amount_ma_', 'amount_std_', 'turnover_',
    )
    name_lower = name.lower()
    if any(name_lower.startswith(p) or p in name_lower for p in _tech_patterns):
        return '技术指标'

    return '其他'


# ─────────────────────────────────────────────────────────────────────────────
# 打印与可视化
# ─────────────────────────────────────────────────────────────────────────────
def print_importance_table(imp: pd.Series, top_n: int, model_name: str, importance_type: str):
    """控制台打印排名表格"""
    total = imp.sum()
    top = imp.head(top_n)

    print(f"\n{'='*70}")
    print(f"  模型: {model_name}   重要性类型: {importance_type}   Top-{top_n}")
    print(f"{'='*70}")
    print(f"  {'排名':<5} {'因子名':<40} {'重要性':>10} {'占比':>8} {'类别'}")
    print(f"  {'-'*65}")

    for rank, (name, val) in enumerate(top.items(), 1):
        pct = val / total * 100 if total > 0 else 0
        category = classify_factor(name)
        # 截断过长的因子名
        display_name = name if len(name) <= 38 else name[:35] + '...'
        print(f"  {rank:<5} {display_name:<40} {val:>10.4f} {pct:>7.2f}%  {category}")

    print(f"  {'-'*65}")
    print(f"  Top-{top_n} 合计占比: {top.sum()/total*100:.1f}%  |  总特征数: {len(imp)}")
    print(f"{'='*70}")


def print_category_summary(imp: pd.Series):
    """按类别汇总重要性"""
    total = imp.sum()
    records = []
    for name, val in imp.items():
        records.append({'factor': name, 'importance': val, 'category': classify_factor(name)})
    df = pd.DataFrame(records)
    summary = df.groupby('category')['importance'].agg(['sum', 'count', 'mean'])
    summary['占比%'] = summary['sum'] / total * 100
    summary = summary.sort_values('sum', ascending=False)
    summary.columns = ['重要性合计', '因子数', '平均重要性', '占比%']

    print(f"\n{'─'*55}")
    print(f"  因子类别汇总")
    print(f"{'─'*55}")
    print(summary.to_string(float_format=lambda x: f'{x:.4f}'))
    print(f"{'─'*55}")


def plot_importance(imp: pd.Series, top_n: int, model_name: str,
                    importance_type: str, output_prefix: str):
    """生成横向条形图并保存"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # 类别颜色映射
        color_map = {
            '技术指标': '#4C72B0',
            'K线形态': '#DD8452',
            '基本面':  '#55A868',
            '市场情绪': '#C44E52',
            '高级时序': '#8172B2',
            '特征工程': '#937860',
            '行业':    '#DA8BC3',
            '状态':    '#8C8C8C',
            '其他':    '#CCCCCC',
        }

        top = imp.head(top_n)
        categories = [classify_factor(n) for n in top.index]
        colors = [color_map.get(c, '#CCCCCC') for c in categories]

        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))
        bars = ax.barh(range(len(top)), top.values, color=colors, edgecolor='white', linewidth=0.5)

        # Y 轴标签
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=9)
        ax.invert_yaxis()

        # 数值标注
        total = imp.sum()
        for i, (val, bar) in enumerate(zip(top.values, bars)):
            pct = val / total * 100 if total > 0 else 0
            ax.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                    f'{pct:.1f}%', va='center', fontsize=8, color='#333333')

        # 图例
        seen = {}
        for cat, col in zip(categories, colors):
            if cat not in seen:
                seen[cat] = col
        patches = [mpatches.Patch(color=c, label=l) for l, c in seen.items()]
        ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.8)

        ax.set_xlabel(f'Feature Importance ({importance_type})', fontsize=10)
        ax.set_title(f'Top-{top_n} Factor Importance — {model_name}', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        out_path = f'{output_prefix}_{model_name}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  图表已保存: {out_path}")
    except ImportError:
        print("  [!] matplotlib 未安装，跳过图表生成")
    except Exception as e:
        print(f"  [!] 图表生成失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  因子重要性分析")
    print(f"  模型路径: {args.model}")
    print(f"  Top-N: {args.top}   重要性类型: {args.type}")
    print(f"{'='*70}\n")

    # 1. 加载模型
    print("正在加载模型...")
    models = load_model(args.model)
    if not models:
        print("未找到可用模型，退出。")
        sys.exit(1)

    all_results = {}

    for model_name, model in models.items():
        print(f"\n处理模型: {model_name}")

        try:
            # 2. 提取重要性
            from core.factors.ml_factor_model import EnsembleFactorModel
            if isinstance(model, EnsembleFactorModel):
                imp = extract_ensemble_importance(model, args.type)
            else:
                imp = extract_importance(model, args.type)

            if imp.empty or imp.sum() == 0:
                print(f"  [!] {model_name} 重要性全为 0，可能模型未训练或格式不支持")
                continue

            all_results[model_name] = imp

            # 3. 打印表格
            print_importance_table(imp, args.top, model_name, args.type)
            print_category_summary(imp)

            # 4. 保存
            if args.save:
                # CSV
                df_out = imp.reset_index()
                df_out.columns = ['factor', 'importance']
                df_out['rank'] = range(1, len(df_out) + 1)
                df_out['category'] = df_out['factor'].apply(classify_factor)
                df_out['importance_pct'] = df_out['importance'] / df_out['importance'].sum() * 100
                csv_path = f'{args.output}_{model_name}.csv'
                df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"  CSV 已保存: {csv_path}")

                # 图表
                plot_importance(imp, args.top, model_name, args.type, args.output)

        except Exception as e:
            import traceback
            print(f"  [X] 处理 {model_name} 失败: {e}")
            traceback.print_exc()

    # 5. 多模型对比（如果有多个模型）
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  多模型 Top-{args.top} 因子对比")
        print(f"{'='*70}")

        # 构建对比 DataFrame
        compare_df = pd.DataFrame(all_results).fillna(0)
        # 每列归一化
        for col in compare_df.columns:
            s = compare_df[col].sum()
            if s > 0:
                compare_df[col] = compare_df[col] / s * 100

        # 按各模型重要性均值排序
        compare_df['mean'] = compare_df.mean(axis=1)
        compare_df = compare_df.sort_values('mean', ascending=False).head(args.top)
        compare_df = compare_df.drop(columns='mean')

        print(compare_df.round(2).to_string())

        if args.save:
            compare_path = f'{args.output}_comparison.csv'
            compare_df.to_csv(compare_path, encoding='utf-8-sig')
            print(f"\n  对比 CSV 已保存: {compare_path}")


if __name__ == '__main__':
    main()
