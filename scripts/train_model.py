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
    parser.add_argument('--workers', type=int, default=15, help='并行线程数')

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
        print(f"\n[Step 0] 检查并增量更新因子缓存 (目标: {cache_end_date})...")
        
        # 优化：预先检查已经是最新的缓存，避免全量加载行情数据到内存
        import pyarrow.parquet as pq
        import sqlite3 as _sqlite3
        cache_dir = TrainingConfig.CACHE_DIR
        stocks_to_update = []
        skipped_count = 0

        # 用数据库中实际最新交易日作为跳过基准，而非"今天"
        # 避免非交易日/盘后运行时，缓存日期永远 < 今天，导致所有缓存被误判为需要更新
        _actual_latest_date = cache_end_date  # 兜底：若查询失败则退回原逻辑
        try:
            _conn = _sqlite3.connect(DATABASE_PATH)
            _row = _conn.execute("SELECT MAX(date) FROM daily_data").fetchone()
            _conn.close()
            if _row and _row[0]:
                _actual_latest_date = str(_row[0])
        except Exception:
            pass
        
        print(f"  正在扫描 {len(trainer_stocks)} 只股票的缓存状态...")
        for code in trainer_stocks:
            cache_file = os.path.join(cache_dir, f'{code}_factors.parquet')
            if os.path.exists(cache_file):
                try:
                    # 快速读取日期列的最后一行
                    last_row = pq.read_table(cache_file, columns=['date']).to_pandas().tail(1)
                    if not last_row.empty:
                        cache_last_date = str(last_row['date'].iloc[0])
                        if cache_last_date >= _actual_latest_date:
                            skipped_count += 1
                            continue
                except Exception:
                    pass
            stocks_to_update.append(code)
            
        print(f"  扫描完成: {skipped_count} 只已同步，{len(stocks_to_update)} 只待更新")
        
        if stocks_to_update:
            # 仅为待更新股票加载数据
            cache_data = trainer.load_training_data(stocks_to_update, train_start_date, cache_end_date)
            
            # 特征发现：采样一小部分（如果是增量，通常已有缓存，逻辑会很快）
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
            print(f"✓ 所有股票缓存均已是最新的 ({cache_end_date})")
            
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
    full_dataset = trainer.prepare_dataset(
        stocks_data,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        include_fundamentals=True,
        n_jobs=args.workers,
        target_features=target_features  # 复用 Step 0 发现的特征集，跳过重复发现
    )

    # ── 7. 训练模型 ──────────────────────────────────────────────────────
    print(f"\n[Step 3] 训练机器学习模型...")
    
    # 解析数据集
    X, y, returns, factor_names, dates, unbuyable_mask, limit_groups = full_dataset[:7]
    path_scores = full_dataset[7] if len(full_dataset) > 7 else None
    is_st_arr = full_dataset[8] if len(full_dataset) > 8 else None
    
    results = trainer.train_models(
        X, y, returns, factor_names, dates,
        unbuyable_mask=unbuyable_mask,
        limit_groups=limit_groups,
        path_scores=path_scores,
        is_st_arr=is_st_arr
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
