import sys
import os
import argparse
import time
from datetime import datetime, timedelta

# 增加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.factors.train_ml_model import (
    FACTOR_CACHE_PREPROCESSING_VERSION,
    MLModelTrainer,
    _read_factor_cache_version,
)
from core.factors.ml_factor_model import MultiObjectiveFactorModel
from config.jydb_config import DATABASE_PATH
from config.factor_config import TrainingConfig, ModelConfig
import pandas as pd
import sqlite3 as _sqlite3


def main():
    default_workers = getattr(TrainingConfig, 'N_JOBS_FACTOR_CALC', 15)

    parser = argparse.ArgumentParser(description='Quantified Decision - 机器学习模型训练入口')
    parser.add_argument('--start',  type=str, default=None, help='训练开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end',    type=str, default=None, help='训练结束日期 (YYYY-MM-DD)')
    parser.add_argument('--stocks', type=int, default=TrainingConfig.STOCK_NUM,
                        help=f'训练选取的股票数量 (默认{TrainingConfig.STOCK_NUM})')
    parser.add_argument('--force',  action='store_true', help='强制重新计算所有因子')
    parser.add_argument('--workers', type=int, default=default_workers, help=f'并行线程数 (默认{default_workers})')
    parser.add_argument('--objective-workers', type=int, default=1,
                        help='多目标训练并行进程数 (默认1=串行; >1 时各目标并行训练, 每进程自动限制 LightGBM 线程避免 oversubscription)')
    parser.add_argument('--model-type', type=str, default='lightgbm', choices=['xgboost', 'lightgbm'],
                        help='基学习器类型: xgboost 或 lightgbm (默认 lightgbm)')

    # ── 增量缓存控制 ──
    parser.add_argument('--update-cache-only', action='store_true',
                        help='仅更新因子缓存到最新日期，不训练模型')
    parser.add_argument('--skip-cache-update', action='store_true',
                        help='跳过增量缓存更新步骤，直接进入模型训练')
    parser.add_argument('--cache-end', type=str, default=None,
                        help='缓存更新截止日期 (YYYY-MM-DD)，默认=今天')
    parser.add_argument('--resume', action='store_true',
                        help='断点续跑：若上次训练被中断(如 OOM 被杀)，载入最新检查点继续，'
                             '只训练未完成目标。需配合同一 latest 目录。')

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
    print(f"不可买入样本处理: {TrainingConfig.UNBUYABLE_HANDLING}")

    trainer = MLModelTrainer(db_path=DATABASE_PATH)

    # ── 3. 获取股票列表（纯聚源：从 daily_data 取 DISTINCT code）───────
    from core.data.jydb_market_etl import JYDBMarketETL
    stock_codes_all = JYDBMarketETL.get_stock_list(
        DATABASE_PATH, as_of_date=cache_end_date, limit=args.stocks
    )
    trainer_stocks = stock_codes_all[:args.stocks]

    # ── 3b. 按策略条件过滤训练股票池（可选）────────────────────────────
    if TrainingConfig.FILTER_BY_STRATEGY:
        import config.strategy_config as sc
        from config import SUPPORTED_MARKETS

        # 确定各过滤参数（TrainingConfig 中的覆盖值优先，None 则回退到 strategy_config）
        filter_markets  = TrainingConfig.TRAIN_FILTER_MARKETS  if TrainingConfig.TRAIN_FILTER_MARKETS  is not None else sc.SELECTOR_MARKETS
        filter_max_price = TrainingConfig.TRAIN_FILTER_MAX_PRICE if TrainingConfig.TRAIN_FILTER_MAX_PRICE is not None else sc.MAX_PRICE
        filter_min_price = TrainingConfig.TRAIN_FILTER_MIN_PRICE if TrainingConfig.TRAIN_FILTER_MIN_PRICE is not None else sc.MIN_PRICE
        filter_include_st = TrainingConfig.TRAIN_FILTER_INCLUDE_ST if TrainingConfig.TRAIN_FILTER_INCLUDE_ST is not None else sc.INCLUDE_ST

        before = len(trainer_stocks)

        # 1. 市场前缀过滤
        if filter_markets:
            allowed_prefixes = []
            for m in filter_markets:
                info = SUPPORTED_MARKETS.get(m, {})
                allowed_prefixes.extend(info.get('prefixes', []))
            if allowed_prefixes:
                trainer_stocks = [c for c in trainer_stocks if c.startswith(tuple(allowed_prefixes))]

        # 2. 价格过滤（需要查数据库最新收盘价）
        if filter_max_price is not None or filter_min_price is not None:
            import sqlite3 as _sqlite3
            _conn = _sqlite3.connect(DATABASE_PATH)
            placeholders = ','.join(['?' for _ in trainer_stocks])
            price_df = pd.read_sql_query(
                f"SELECT code, close FROM daily_data WHERE code IN ({placeholders})"
                f" AND date = (SELECT MAX(date) FROM daily_data)",
                _conn, params=trainer_stocks
            )
            _conn.close()
            price_map = dict(zip(price_df['code'], price_df['close']))
            filtered = []
            for c in trainer_stocks:
                p = price_map.get(c)
                if p is None:
                    filtered.append(c)  # 无价格数据的保留（历史样本）
                    continue
                if filter_min_price is not None and p < filter_min_price:
                    continue
                if filter_max_price is not None and p > filter_max_price:
                    continue
                filtered.append(c)
            trainer_stocks = filtered

        # 3. ST 过滤（排除当前为 ST 的股票）
        if not filter_include_st:
            import sqlite3 as _sqlite3
            _conn = _sqlite3.connect(DATABASE_PATH)
            placeholders = ','.join(['?' for _ in trainer_stocks])
            st_df = pd.read_sql_query(
                f"SELECT code, is_st FROM daily_data WHERE code IN ({placeholders})"
                f" AND date = (SELECT MAX(date) FROM daily_data)",
                _conn, params=trainer_stocks
            )
            _conn.close()
            st_map = dict(zip(st_df['code'], st_df['is_st']))
            trainer_stocks = [c for c in trainer_stocks if st_map.get(c, 0) != 1]

        after = len(trainer_stocks)
        print(f"[策略过滤] 股票池: {before} → {after} 只"
              f" (市场={filter_markets}, 价格=[{filter_min_price},{filter_max_price}], ST={filter_include_st})")
    else:
        print(f"[策略过滤] 已关闭，使用全市场 {len(trainer_stocks)} 只股票训练")

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
        
        def _scan_one_cache(code):
            cache_file = os.path.join(cache_dir, f'{code}_factors.parquet')
            if os.path.exists(cache_file):
                try:
                    if (
                        _read_factor_cache_version(cache_file)
                        != FACTOR_CACHE_PREPROCESSING_VERSION
                    ):
                        return code, True
                    pf = pq.ParquetFile(cache_file)
                    if pf.num_row_groups > 0:
                        table = pf.read_row_group(pf.num_row_groups - 1, columns=['date'])
                    else:
                        table = pq.read_table(cache_file, columns=['date'])
                    if table.num_rows > 0:
                        cache_last_date = str(table.column('date')[-1].as_py())[:10]
                        return code, cache_last_date < _actual_latest_date
                except Exception:
                    pass
            return code, True

        print(f"  正在并行扫描 {len(trainer_stocks)} 只股票的缓存状态...")
        from concurrent.futures import ThreadPoolExecutor
        scan_workers = max(1, min(32, args.workers if args.workers and args.workers > 0 else 32, len(trainer_stocks)))
        with ThreadPoolExecutor(max_workers=scan_workers) as executor:
            for code, needs_update in executor.map(_scan_one_cache, trainer_stocks):
                if needs_update:
                    stocks_to_update.append(code)
                else:
                    skipped_count += 1
            
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
            print(f"[OK] 所有股票缓存均已是最新的 ({cache_end_date})")
            
    else:
        print("\n[Step 0] 跳过增量缓存更新 (--skip-cache-update)")

    # 如果只做缓存更新，到此退出
    if args.update_cache_only:
        print("\n=== 因子缓存更新流程完成 ===")
        return

    # ── 5. 加载训练数据 ──────────────────────────────────────────────────
    print(f"\n[Step 1] 正在读取训练标签行情数据...")
    stocks_data = trainer.load_label_data(trainer_stocks, train_start_date, train_end_date)

    # ── 6. 准备数据集 ────────────────────────────────────────────────────
    print(f"\n[Step 2] 准备特征数据集与标签...")
    print("  正在构造多目标标签...")
    multi_labels = trainer.build_multiobjective_labels(
        stocks_data, train_start_date, train_end_date
    )

    full_dataset = trainer.prepare_dataset(
        stocks_data,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        include_fundamentals=True,
        n_jobs=args.workers,
        target_features=target_features,  # 复用 Step 0 发现的特征集，跳过重复发现
        use_factor_cache_only=True,
        return_codes=True,
    )
    del stocks_data
    import gc; gc.collect()

    # ── 7. 训练模型 ──────────────────────────────────────────────────────
    print(f"\n[Step 3] 训练多目标机器学习模型...")

    X, y, returns, factor_names, dates, unbuyable_mask, limit_groups, path_scores, is_st_arr, w_sig_arr, codes = full_dataset
    keys = pd.DataFrame({
        'date': pd.Series(dates).astype(str).str[:10],
        'code': pd.Series(codes).astype(str),
        '__row_order': range(len(dates)),
    })
    label_frame = multi_labels.copy()
    label_frame['date'] = label_frame['date'].astype(str).str[:10]
    label_frame['code'] = label_frame['code'].astype(str)
    aligned = keys.merge(label_frame, on=['date', 'code'], how='left', sort=False)
    aligned = aligned.sort_values('__row_order').reset_index(drop=True)

    # 断点续跑：检查点路径放在 latest 目录；若 --resume 且检查点存在则载入已有部分模型
    latest_dir = os.path.join(TrainingConfig.SAVE_DIR, 'latest')
    os.makedirs(latest_dir, exist_ok=True)
    checkpoint_path = os.path.join(latest_dir, '.train_checkpoint.pkl')
    resume_from = (MultiObjectiveFactorModel.load_model(checkpoint_path)
                   if (args.resume and os.path.exists(checkpoint_path)) else None)
    if resume_from is not None:
        print(f"  [断点续跑] 载入检查点: {checkpoint_path} "
              f"({len(resume_from.models)} 个已完成目标)")

    multi_model, results, selected_names = trainer.train_multiobjective_models(
        X, aligned, factor_names, dates,
        objective_weights=TrainingConfig.get_objective_weights(),
        model_type=args.model_type,
        objective_workers=args.objective_workers,
        resume_from=resume_from,
        checkpoint_path=checkpoint_path,
    )
    import json, pickle, shutil
    archive_dir = os.path.join(
        TrainingConfig.SAVE_DIR,
        'multi_objective_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    os.makedirs(archive_dir, exist_ok=True)
    archive_model = os.path.join(archive_dir, 'multi_objective_factor_model.pkl')
    latest_model = os.path.join(latest_dir, 'multi_objective_factor_model.pkl')
    multi_model.save_model(archive_model)
    shutil.copy2(archive_model, latest_model)
    with open(os.path.join(archive_dir, 'multi_objective_config.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'objectives': list(multi_model.models),
            'weights': multi_model.weights,
            'selected_features': selected_names,
            'train_start_date': train_start_date,
            'train_end_date': train_end_date,
            'model_version': os.getenv('QD_MODEL_VERSION', 'v1'),
            'model_params': ModelConfig.get_model_params(args.model_type, TrainingConfig.TASK),
        }, f, ensure_ascii=False, indent=2)
    if getattr(trainer, 'norm_stats', None) is not None:
        for target_dir in (archive_dir, latest_dir):
            with open(os.path.join(target_dir, 'norm_stats.pkl'), 'wb') as f:
                pickle.dump(trainer.norm_stats, f)
    trainer.save_factor_summary(selected_names, save_dir=archive_dir)
    print(f"\n=== 多目标模型训练完成: {archive_model} ===")


if __name__ == "__main__":
    main()
