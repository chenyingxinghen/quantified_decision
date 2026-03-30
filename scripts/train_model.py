import sys
import os
import argparse
import time
from datetime import datetime, timedelta

# 增加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.factors.train_ml_model import MLModelTrainer
from config.baostock_config import DATABASE_PATH
from config.factor_config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(description='Quantified Decision - 机器学习模型训练入口')
    parser.add_argument('--start',  type=str, default=None, help='训练开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end',    type=str, default=None, help='训练结束日期 (YYYY-MM-DD)')
    parser.add_argument('--stocks', type=int, default=TrainingConfig.STOCK_NUM,
                        help=f'训练选取的股票数量 (默认{TrainingConfig.STOCK_NUM})')
    parser.add_argument('--force',  action='store_true', help='强制重新计算所有因子')
    parser.add_argument('--workers', type=int, default=12, help='并行线程数')

    # ── 增量缓存控制 ──
    parser.add_argument('--update-cache-only', action='store_true',
                        help='仅更新因子缓存到最新日期，不训练模型')
    parser.add_argument('--skip-cache-update', action='store_true',
                        help='跳过增量缓存更新步骤，直接进入模型训练')
    parser.add_argument('--cache-end', type=str, default=None,
                        help='缓存更新截止日期 (YYYY-MM-DD)，默认=今天')

    args = parser.parse_args()

    # ── 1. 自动设置训练日期范围 ───────────────────────────────────────────
    train_end_date = (
        args.end if args.end
        else (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)).strftime('%Y-%m-%d')
    )
    train_start_date = (
        args.start if args.start
        else (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_TRAINING)).strftime('%Y-%m-%d')
    )

    # 缓存更新截止日期：默认为"今天"（捕获最新行情）
    cache_end_date = args.cache_end if args.cache_end else datetime.now().strftime('%Y-%m-%d')

    print(f"=== 模型训练启动 ===")
    print(f"训练数据范围: {train_start_date} 至 {train_end_date}")
    print(f"缓存更新截止: {train_start_date} 至 {cache_end_date}")
    print(f"股票样本: 前 {args.stocks} 只")
    print(f"涨停板/停牌样本惩罚加权: {'是' if TrainingConfig.PUNISH_UNBUYABLE else '否'}")

    # ── 2. 初始化训练器 ──────────────────────────────────────────────────
    trainer = MLModelTrainer(db_path=DATABASE_PATH, punish_unbuyable=TrainingConfig.PUNISH_UNBUYABLE)

    # ── 3. 获取股票列表 ──────────────────────────────────────────────────
    from core.data.baostock_main import BaostockDataManager
    manager = BaostockDataManager()
    stock_list = manager.get_stock_list_from_db()
    trainer_stocks = stock_list['code'].tolist()[:args.stocks]
    manager.close()

    # ── 4. 增量更新因子缓存（到最新日期）────────────────────────────────
    target_features = None  # 特征集，Step 0 发现后供 Step 2 复用
    if not args.skip_cache_update:
        print(f"\n[Step 0] 增量更新因子缓存 (截止 {cache_end_date})...")
        # 加载数据，用于更新缓存
        cache_data = trainer.load_training_data(trainer_stocks, train_start_date, cache_end_date)
        # 先发现完整特征集，再用同一套 target_features 写缓存
        # 这样 Step 2 的缓存命中检查能与缓存列完全匹配，避免全量重算
        target_features = trainer.discover_target_features(
            cache_data, include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS
        )
        trainer.batch_update_factor_cache(
            stocks_data=cache_data,
            include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
            target_features=target_features,
            n_jobs=args.workers
        )
        del cache_data  # 释放内存
        import gc; gc.collect()
    else:
        print("\n[Step 0] 跳过增量缓存更新 (--skip-cache-update)")

    # 如果只做缓存更新，到此退出
    if args.update_cache_only:
        print("\n=== 因子缓存更新流程完成 ===")
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
        include_fundamentals=True,
        n_jobs=args.workers,
        target_features=target_features  # 复用 Step 0 发现的特征集，跳过重复发现
    )

    # ── 7. 训练模型 ──────────────────────────────────────────────────────
    print(f"\n[Step 3] 训练机器学习模型...")
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

    print("\n保存最新模型...")
    archive_dir = trainer.save_models(
        save_dir=TrainingConfig.SAVE_DIR,
        years=TrainingConfig.YEARS_FOR_TRAINING,
        stocks=len(trainer_stocks)
    )

    trainer.save_factor_summary(factor_names, save_dir=archive_dir)

    print(f"\n=== 流程结束: 模型已就绪 ===")


if __name__ == "__main__":
    main()
