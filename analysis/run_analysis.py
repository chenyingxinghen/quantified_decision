"""
analysis/run_analysis.py — 分析编排入口（CLI）

用法:
  # 本地用真实模型 + 合成数据冒烟测试（无需数据库）
  python -m analysis.run_analysis --synthetic --model models/latest/multi_objective_factor_model.pkl

  # 云端：用真实数据库构建 full_dataset 并完整分析
  python -m analysis.run_analysis \
      --model models/latest/multi_objective_factor_model.pkl \
      --out analysis/output --stocks 2000 --cache analysis/full_dataset.parquet

输出:
  <out>/report.md, <out>/metrics.csv, <out>/figures/*.png
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from datetime import datetime

import numpy as np

# 项目根加入路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from analysis.common import (  # noqa: E402
    load_model, normalize_features, predict_scores, classify_regimes,
    build_full_dataset, make_synthetic_dataset,
)
from analysis import robustness as rob  # noqa: E402
from analysis import portfolio as port  # noqa: E402
from analysis import shap_analysis as shapmod  # noqa: E402
from analysis import mechanism as mech  # noqa: E402
from analysis import report as rep  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="量化因子模型 稳健性检验与拓展分析")
    ap.add_argument("--model", default=os.path.join("models", "latest",
                                                     "multi_objective_factor_model.pkl"))
    ap.add_argument("--dataset", default=None,
                    help="full_dataset parquet 缓存；缺失则自动构建/合成")
    ap.add_argument("--out", default=os.path.join("analysis", "output"))
    ap.add_argument("--stocks", type=int, default=2000,
                    help="构建数据集使用的股票数（云端建议全量）")
    ap.add_argument("--cache", default=os.path.join("analysis", "full_dataset.parquet"),
                    help="数据集 parquet 缓存路径")
    ap.add_argument("--synthetic", action="store_true",
                    help="使用与模型维度一致的合成数据（无数据库环境冒烟测试）")
    ap.add_argument("--q", type=int, default=10, help="分组数量")
    ap.add_argument("--cost", type=float, default=0.001, help="单边交易成本")
    ap.add_argument("--shap-sample", type=int, default=5000,
                    help="SHAP 计算采样行数（控制耗时）")
    ap.add_argument("--no-shap", action="store_true",
                    help="跳过 SHAP 分析（神经网络模型无树结构，SHAP 不适用时可开启）")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    print(f"[analysis] 加载模型: {args.model}")
    model = load_model(args.model)
    # 神经网络模型没有树结构，SHAP TreeExplainer 不适用；--no-shap 或检测到
    # 神经网络模型时跳过 SHAP，分析报告将改用首层权重近似重要性说明。
    is_neural = type(model).__name__ == "MultiObjectiveNeuralModel"
    do_shap = (not args.no_shap) and (not is_neural)
    if is_neural:
        print("[analysis] 检测到神经网络模型：SHAP 不适用，已自动跳过 SHAP 分析")

    # ── 1. 获取数据 ──
    if args.synthetic:
        print("[analysis] 合成数据模式（仅验证代码路径）")
        fd = make_synthetic_dataset(model, n_samples=max(5000, args.shap_sample * 2))
    elif args.dataset and os.path.exists(args.dataset):
        # 缓存失效检查：模型 pkl mtime 比缓存目录新 → 缓存里的 feature_names 可能
        # 与当前模型不一致（典型：v1 训练后改代码 → v2 训练 → 缓存还是 v1 时期建的，
        # SHAP 取交集时会因 'is_suspended' 等列漂移而崩）。自动重建。
        cache_root = (args.dataset[:-len(".parquet")]
                      if args.dataset.endswith(".parquet") else args.dataset)
        cache_meta = os.path.join(cache_root, "meta.pkl")
        cache_mtime = os.path.getmtime(cache_meta) if os.path.exists(cache_meta) else 0.0
        model_mtime = os.path.getmtime(args.model) if os.path.exists(args.model) else 0.0
        if model_mtime > cache_mtime + 1.0:
            print(f"[dataset] 模型比缓存新（{datetime.fromtimestamp(model_mtime):%Y-%m-%d %H:%M:%S} "
                  f"> {datetime.fromtimestamp(cache_mtime):%Y-%m-%d %H:%M:%S}），自动重建缓存")
            import shutil as _sh
            _sh.rmtree(cache_root, ignore_errors=True)
            from core.factors.train_ml_model import MLModelTrainer as _TR
            from config.jydb_config import DATABASE_PATH as _DB
            _trainer = _TR(db_path=_DB)
            fd = build_full_dataset(_trainer, start_date=args.start, end_date=args.end,
                                    stocks=args.stocks, cache_path=args.cache)
        else:
            from analysis.common import load_full_dataset_from_parquet
            fd = load_full_dataset_from_parquet(args.dataset)
    else:
        from core.factors.train_ml_model import MLModelTrainer
        from config.jydb_config import DATABASE_PATH
        trainer = MLModelTrainer(db_path=DATABASE_PATH)
        fd = build_full_dataset(trainer, start_date=args.start, end_date=args.end,
                                stocks=args.stocks, cache_path=args.cache)

    (X, y, returns, factor_names, dates, unbuyable_mask,
     limit_groups, path_scores, is_st_arr, w_sig_arr, codes) = fd
    print(f"[analysis] 样本: X={X.shape}, 特征数={len(factor_names)}")
    n_samples = X.shape[0]  # 在 del X 之前保存样本数，供末尾 results meta 使用

    # ── 2. 归一化 + 预测 ──
    from core.factors.train_ml_model import MLModelTrainer
    from config.jydb_config import DATABASE_PATH
    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    trainer.norm_stats = None
    Xn = normalize_features(trainer, X, dates, factor_names)
    scores = predict_scores(model, Xn, factor_names)
    # X 之后不再需要（下游模块只用 scores/returns/dates/codes），立即释放 ~9.5GB
    del X
    gc.collect()

    # ── 3. 市场状态 ──
    regime_per_sample, market_daily, date_to_regime = classify_regimes(dates, returns)

    # ── 4. 各分析模块 ──
    print("[analysis] 统计性检验 + 牛熊震荡异质性 ...")
    robustness_res = rob.run_robustness(scores, returns, dates, regime_per_sample)

    print("[analysis] 投资组合检验（分组/多空/换手/成本）...")
    portfolio_res = port.run_portfolio(scores, returns, dates, codes,
                                       q=args.q, cost_per_trade=args.cost)

    print("[analysis] SHAP 分析（全局重要性/类别贡献/交互/一致性）...")
    if do_shap:
        shap_res = shapmod.run_shap(model, Xn, factor_names)
    else:
        # 神经网络模型（无树结构）或 --no-shap：构造占位结构，供机制分析与报告一致消费
        reason = "neural" if is_neural else "disabled"
        shap_res = {
            "skipped": True,
            "reason": reason,
            "global": {"method": "neural-skip" if is_neural else "disabled", "importance": {}},
            "category": {"method": "skipped", "share": {}, "shares": {}},
            "interaction": {"skipped": True},
            "consistency": {"skipped": True, "per_objective_rank_corr": {},
                            "mean_consistency": float("nan")},
        }

    # SHAP 已完成（或跳过），释放归一化特征矩阵 ~9.5GB
    del Xn
    gc.collect()

    print("[analysis] 机制分析 ...")
    mechanism_res = mech.build_mechanism_narrative({
        "robustness": robustness_res,
        "portfolio": portfolio_res,
        "shap": shap_res,
    })

    results = {
        "robustness": robustness_res,
        "portfolio": portfolio_res,
        "shap": shap_res,
        "mechanism": mechanism_res,
        "meta": {
            "n_samples": int(n_samples),
            "train_range": f"{str(dates[0])[:10]}~{str(dates[-1])[:10]}",
        },
    }

    # ── 5. 报告 + 图表 ──
    print(f"[analysis] 生成报告 -> {args.out}")
    out = rep.write_report(results, args.out)
    print("[analysis] 完成:")
    for k, v in out.items():
        if k != "figures":
            print(f"  - {k}: {v}")
    print(f"  - 图表目录: {os.path.join(args.out, 'figures')}")


if __name__ == "__main__":
    main()
