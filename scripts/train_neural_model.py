"""
Quantified Decision - 神经网络模型训练入口

与 scripts/train_model.py 平行，但用 PyTorch MLP 替代 LightGBM/XGBoost。
完全复用同一套数据基础设施：
- 因子 parquet 缓存（增量更新到最新交易日）
- 多目标标签（build_multiobjective_labels）
- 横截面归一化与特征选择（与 GBM 完全一致）
- 模型保存路径与 norm_stats.pkl（可直接被现有回测/实盘选股消费）
"""
import sys
import os
import argparse
import time
from datetime import datetime, timedelta

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.neural.trainer import NeuralTrainer
from config.jydb_config import DATABASE_PATH
from config.factor_config import TrainingConfig
from config import neural_config as nc
from core.data.jydb_market_etl import JYDBMarketETL
from core.factors.train_ml_model import (
    FACTOR_CACHE_PREPROCESSING_VERSION,
    _read_factor_cache_version,
)

import pyarrow.parquet as pq
import sqlite3 as _sqlite3
import pandas as pd


def _scan_stocks_needing_update(trainer_stocks, cache_dir, actual_latest_date, workers):
    def _scan_one(code):
        cache_file = os.path.join(cache_dir, f"{code}_factors.parquet")
        if os.path.exists(cache_file):
            try:
                if _read_factor_cache_version(cache_file) != FACTOR_CACHE_PREPROCESSING_VERSION:
                    return code, True
                pf = pq.ParquetFile(cache_file)
                if pf.num_row_groups > 0:
                    table = pf.read_row_group(pf.num_row_groups - 1, columns=["date"])
                else:
                    table = pq.read_table(cache_file, columns=["date"])
                if table.num_rows > 0:
                    last = str(table.column("date")[-1].as_py())[:10]
                    return code, last < actual_latest_date
            except Exception:
                pass
        return code, True

    from concurrent.futures import ThreadPoolExecutor
    need = []
    skip = 0
    w = max(1, min(32, workers or 32, len(trainer_stocks)))
    with ThreadPoolExecutor(max_workers=w) as ex:
        for code, upd in ex.map(_scan_one, trainer_stocks):
            if upd:
                need.append(code)
            else:
                skip += 1
    return need, skip


def main():
    default_workers = getattr(TrainingConfig, "N_JOBS_FACTOR_CALC", 15)
    parser = argparse.ArgumentParser(description="量化决策 - 神经网络模型训练")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--stocks", type=int, default=TrainingConfig.STOCK_NUM)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--force", action="store_true", help="强制重算所有因子缓存")
    parser.add_argument("--update-cache-only", action="store_true")
    parser.add_argument("--skip-cache-update", action="store_true")
    parser.add_argument("--cache-end", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=nc.NeuralConfig.EPOCHS,
                        help=f"训练轮数 (默认 {nc.NeuralConfig.EPOCHS})")
    parser.add_argument("--save-dir", type=str, default=TrainingConfig.SAVE_DIR)
    args = parser.parse_args()

    train_end_date = (args.end or
                      (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)).strftime("%Y-%m-%d"))
    train_start_date = (args.start or
                        (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_TRAINING)).strftime("%Y-%m-%d"))
    cache_end_date = args.cache_end or datetime.now().strftime("%Y-%m-%d")

    print("=== 神经网络模型训练启动 ===")
    print(f"训练数据范围: {train_start_date} 至 {train_end_date}")
    print(f"缓存更新截止: {train_start_date} 至 {cache_end_date}")

    trainer = NeuralTrainer(db_path=DATABASE_PATH, neural_cfg=nc.NeuralConfig)

    # 1. 股票列表（纯聚源：从 daily_data 取 DISTINCT code）
    stock_codes_all = JYDBMarketETL(DATABASE_PATH).get_stock_list(
        DATABASE_PATH, as_of_date=cache_end_date, limit=args.stocks
    )
    trainer_stocks = stock_codes_all[:args.stocks]

    # 2. 策略过滤（与 train_model.py 一致，可选）
    if TrainingConfig.FILTER_BY_STRATEGY:
        import config.strategy_config as sc
        from config import SUPPORTED_MARKETS
        filter_markets = TrainingConfig.TRAIN_FILTER_MARKETS if TrainingConfig.TRAIN_FILTER_MARKETS is not None else sc.SELECTOR_MARKETS
        filter_max_price = TrainingConfig.TRAIN_FILTER_MAX_PRICE if TrainingConfig.TRAIN_FILTER_MAX_PRICE is not None else sc.MAX_PRICE
        filter_min_price = TrainingConfig.TRAIN_FILTER_MIN_PRICE if TrainingConfig.TRAIN_FILTER_MIN_PRICE is not None else sc.MIN_PRICE
        filter_include_st = TrainingConfig.TRAIN_FILTER_INCLUDE_ST if TrainingConfig.TRAIN_FILTER_INCLUDE_ST is not None else sc.INCLUDE_ST
        before = len(trainer_stocks)
        if filter_markets:
            prefixes = []
            for m in filter_markets:
                prefixes.extend(SUPPORTED_MARKETS.get(m, {}).get("prefixes", []))
            if prefixes:
                trainer_stocks = [c for c in trainer_stocks if c.startswith(tuple(prefixes))]
        if filter_max_price is not None or filter_min_price is not None:
            _conn = _sqlite3.connect(DATABASE_PATH)
            ph = ",".join(["?"] * len(trainer_stocks))
            pdf = pd.read_sql_query(f"SELECT code, close FROM daily_data WHERE code IN ({ph}) AND date=(SELECT MAX(date) FROM daily_data)", _conn, params=trainer_stocks)
            _conn.close()
            pm = dict(zip(pdf["code"], pdf["close"]))
            trainer_stocks = [c for c in trainer_stocks if pm.get(c) is not None and (filter_min_price is None or pm[c] >= filter_min_price) and (filter_max_price is None or pm[c] <= filter_max_price)]
        if not filter_include_st:
            _conn = _sqlite3.connect(DATABASE_PATH)
            ph = ",".join(["?"] * len(trainer_stocks))
            sdf = pd.read_sql_query(f"SELECT code, is_st FROM daily_data WHERE code IN ({ph}) AND date=(SELECT MAX(date) FROM daily_data)", _conn, params=trainer_stocks)
            _conn.close()
            sm = dict(zip(sdf["code"], sdf["is_st"]))
            trainer_stocks = [c for c in trainer_stocks if sm.get(c, 0) != 1]
        print(f"[策略过滤] 股票池: {before} → {len(trainer_stocks)} 只")
    else:
        print(f"[策略过滤] 已关闭，使用全市场 {len(trainer_stocks)} 只股票")

    # 3. 增量更新因子缓存（与 GBM 共用同一份 parquet）
    target_features = None
    if not args.skip_cache_update:
        print(f"\n[NN-Step 0] 增量更新因子缓存 (目标: {cache_end_date})...")
        actual_latest = cache_end_date
        try:
            _c = _sqlite3.connect(DATABASE_PATH)
            _r = _c.execute("SELECT MAX(date) FROM daily_data").fetchone()
            _c.close()
            if _r and _r[0]:
                actual_latest = str(_r[0])
        except Exception:
            pass
        need, skipped = _scan_stocks_needing_update(
            trainer_stocks, TrainingConfig.CACHE_DIR, actual_latest, args.workers
        )
        print(f"  扫描完成: {skipped} 只已同步，{len(need)} 只待更新")
        if args.force:
            need = trainer_stocks
        if need:
            cache_data = trainer._trainer.load_training_data(need, train_start_date, cache_end_date)
            target_features = trainer._trainer.discover_target_features(
                cache_data, include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS
            )
            trainer._trainer.batch_update_factor_cache(
                stocks_data=cache_data,
                include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
                target_features=target_features,
                n_jobs=args.workers,
            )
            import gc; gc.collect()
        else:
            print(f"[OK] 所有股票缓存均已是最新 ({cache_end_date})")
    else:
        print("\n[NN-Step 0] 跳过增量缓存更新 (--skip-cache-update)")

    if args.update_cache_only:
        print("\n=== 因子缓存更新流程完成 ===")
        return

    # 4. 构建数据集（复用 MLModelTrainer）
    dataset = trainer.build_dataset(
        trainer_stocks, train_start_date, train_end_date,
        target_features=target_features, workers=args.workers,
    )

    # 5. 多目标神经网络训练
    print(f"\n[NN-Step 4] 训练多目标神经网络模型...")
    model_kwargs = nc.NeuralConfig.to_model_kwargs()
    if args.epochs != nc.NeuralConfig.EPOCHS:
        model_kwargs["epochs"] = args.epochs
    # 注意：神经网络使用独立的 NeuralConfig 多目标权重（强调风险调整收益），
    # 与 GBM 的 TrainingConfig.MULTI_OBJECTIVE_WEIGHTS 不同；build_dataset 生成的
    # 标签列也来自 NeuralConfig，因此这里必须传 NeuralConfig 权重，否则会出现
    # 标签列与权重键不匹配、sharpe 类核心目标被静默丢弃的问题。
    multi_model, selected_names, norm_stats, results = trainer.train_multiobjective(
        dataset, objective_weights=nc.NeuralConfig.MULTI_OBJECTIVE_WEIGHTS,
        model_kwargs=model_kwargs,
    )

    # 6. 保存产物
    archive_dir = trainer.save_artifacts(
        multi_model, selected_names, args.save_dir, norm_stats
    )
    print(f"\n=== 神经网络模型训练完成 -> {archive_dir} ===")


if __name__ == "__main__":
    main()
