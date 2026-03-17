
import sys
import os
import argparse
from datetime import datetime, timedelta

# 增加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.factors.train_ml_model import MLModelTrainer
from core.data.data_fetcher import DataFetcher
from config.config import DATABASE_PATH, YEARS
from config.factor_config import TrainingConfig


def update_factor_cache(trainer: MLModelTrainer,
                        stocks_data: dict,
                        include_fundamentals: bool = True,
                        n_jobs: int = 12):
    """
    增量更新因子缓存：与训练数据范围无关，总是把缓存补齐到行情最新日期。
    """
    from joblib import Parallel, delayed

    stock_list = list(stocks_data.items())
    total = len(stock_list)
    print(f"\n[增量缓存更新] 共 {total} 只股票，使用 {n_jobs} 核心并行...")

    # 特征集发现（用于保证新行列对齐）
    target_features = None
    cache_files = [f for f in os.listdir(trainer.factors_cache_dir) if f.endswith('.parquet')]
    if cache_files:
        import pandas as pd
        try:
            sample = pd.read_parquet(
                os.path.join(trainer.factors_cache_dir, cache_files[0])
            )
            # 取出已有缓存的列名作为目标特征（含 date）
            target_features = [c for c in sample.columns if c != 'date'] or None
        except Exception:
            pass

    def _update_one(code, data):
        try:
            trainer.calculate_and_save_factors(
                code=code,
                data=data,
                apply_feature_engineering=True,
                target_features=target_features,
                verbose=False,
                include_fundamentals=include_fundamentals,
            )
            return 'ok'
        except Exception as e:
            return f'err:{e}'

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_update_one)(code, data) for code, data in stock_list
    )

    ok_count  = sum(1 for r in results if r == 'ok')
    err_count = sum(1 for r in results if r != 'ok')
    print(f"[增量缓存更新] 完成: {ok_count} 只成功, {err_count} 只失败\n")


def main():
    parser = argparse.ArgumentParser(description='Quantified Decision - 机器学习模型训练入口')
    parser.add_argument('--start',  type=str, default=None, help='训练开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end',    type=str, default=None, help='训练结束日期 (YYYY-MM-DD)')
    parser.add_argument('--stocks', type=int, default=TrainingConfig.STOCK_NUM,
                        help=f'训练选取的股票数量 (默认{TrainingConfig.STOCK_NUM})')
    parser.add_argument('--force',  action='store_true', help='强制重新计算所有因子')

    # ── 增量缓存控制 ──
    parser.add_argument('--update-cache-only', action='store_true',
                        help='仅更新因子缓存到最新日期，不训练模型')
    parser.add_argument('--skip-cache-update', action='store_true',
                        help='跳过增量缓存更新步骤，直接进入模型训练')
    parser.add_argument('--cache-end', type=str, default=None,
                        help='缓存更新截止日期 (YYYY-MM-DD)，默认=今天')
    parser.add_argument('--use-amount-turnover', action='store_true',
                        help='是否使用amount和turnover_rate作为特征')

    args = parser.parse_args()

    if args.use_amount_turnover:
        TrainingConfig.USE_AMOUNT_TURNOVER = True

    # ── 1. 自动设置训练日期范围 ───────────────────────────────────────────
    train_end_date = (
        args.end if args.end
        else (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)).strftime('%Y-%m-%d')
    )
    train_start_date = (
        args.start if args.start
        else (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS)).strftime('%Y-%m-%d')
    )

    # 缓存更新截止日期：默认为"今天"（捕获最新行情）
    cache_end_date = args.cache_end if args.cache_end else datetime.now().strftime('%Y-%m-%d')

    print(f"=== 模型训练启动 ===")
    print(f"训练数据范围: {train_start_date} 至 {train_end_date}")
    print(f"缓存更新截止: {train_start_date} 至 {cache_end_date}")
    print(f"股票样本: 前 {args.stocks} 只")
    print(f"涨停板/停牌样本惩罚加权: {'是' if TrainingConfig.PUNISH_UNBUYABLE else '否'}")

    # ── 2. 获取股票列表 ──────────────────────────────────────────────────
    fetcher = DataFetcher()
    stock_list = fetcher.get_stock_list()
    trainer_stocks = stock_list['code'].tolist()[:args.stocks]
    fetcher.close()

    # ── 3. 初始化训练器 ──────────────────────────────────────────────────
    trainer = MLModelTrainer(db_path=DATABASE_PATH, punish_unbuyable=TrainingConfig.PUNISH_UNBUYABLE)

    # ── 4. 增量更新因子缓存（到最新日期）────────────────────────────────
    if not args.skip_cache_update:
        print(f"\n[Step 0] 增量更新因子缓存 (截止 {cache_end_date})...")
        # 加载比训练多出的"最新行情"数据，用于更新缓存
        cache_data = trainer.load_training_data(trainer_stocks, train_start_date, cache_end_date)
        update_factor_cache(
            trainer=trainer,
            stocks_data=cache_data,
            include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
            n_jobs=12
        )
        del cache_data  # 释放内存
        import gc; gc.collect()
    else:
        print("\n[Step 0] 跳过增量缓存更新 (--skip-cache-update)")

    # 如果只做缓存更新，到此退出
    if args.update_cache_only:
        print("\n=== 因子缓存更新完成（--update-cache-only 模式，跳过训练）===")
        return

    # ── 5. 加载训练数据 ──────────────────────────────────────────────────
    print(f"\n[Step 1] 正在读取训练历史行情数据...")
    stocks_data = trainer.load_training_data(trainer_stocks, train_start_date, train_end_date)

    # ── 6. 准备数据集 ────────────────────────────────────────────────────
    print(f"\n[Step 2] 准备特征数据集与标签...")
    X, y, returns, factor_names, dates, unbuyable_mask, limit_groups = trainer.prepare_dataset(
        stocks_data,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        filter_incomplete_cache=not args.force,
        include_fundamentals=True
    )

    # ── 7. 训练模型 ──────────────────────────────────────────────────────
    print(f"\n[Step 3] 训练多种机器学习模型并对比评分...")
    results = trainer.train_models(
        X, y, returns, factor_names, dates,
        unbuyable_mask=unbuyable_mask,
        limit_groups=limit_groups
    )

    # ── 8. 对比与保存 ────────────────────────────────────────────────────
    best_model_type = trainer.compare_models(results)

    if best_model_type is None:
        print("\n[错误] 模型训练全部失败，请检查数据或参数设置。")
        return

    print("\n保存模型...")
    archive_dir = trainer.save_models(
        save_dir=TrainingConfig.SAVE_DIR,
        years=TrainingConfig.YEARS_FOR_TRAINING,
        stocks=len(trainer_stocks)
    )

    trainer.save_factor_summary(factor_names, save_dir=archive_dir)

    print(f"\n=== 训练流程全部完成 ===")


if __name__ == "__main__":
    main()
