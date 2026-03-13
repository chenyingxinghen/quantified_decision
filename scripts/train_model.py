
import sys
import os
import argparse
from datetime import datetime

# 增加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.factors.train_ml_model import MLModelTrainer
from core.data.data_fetcher import DataFetcher
from config.config import DATABASE_PATH, YEARS
from config.factor_config import TrainingConfig

def main():
    parser = argparse.ArgumentParser(description='Quantified Decision - 机器学习模型训练入口')
    parser.add_argument('--start', type=str, default=None, help='训练开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='训练结束日期 (YYYY-MM-DD)')
    parser.add_argument('--stocks', type=int, default=TrainingConfig.STOCK_NUM, help=f'训练选取的股票数量 (默认{TrainingConfig.STOCK_NUM})')
    parser.add_argument('--force', action='store_true', help='强制重新计算所有因子')
    
    args = parser.parse_args()
    
    # 1. 自动设置日期范围 (基于配置中的 训练年数 + 回测年数)
    end_date = args.end if args.end else datetime.now().strftime('%Y-%m-%d')
    if args.start:
        start_date = args.start
    else:
        # 综合考虑训练和回测所需的总时长 (例如 12年训练 + 3年回测 = 15年数据)
        start_year = datetime.now().year - YEARS - 1
        start_date = f"{start_year}-01-01"
        
    print(f"=== 模型训练启动 ===")
    print(f"数据范围: {start_date} 至 {end_date}")
    print(f"股票样本: 前 {args.stocks} 只")
    print(f"涨停板/停牌样本惩罚加权: {'是' if TrainingConfig.PUNISH_UNBUYABLE else '否'}")
    
    # 2. 获取股票列表
    fetcher = DataFetcher()
    stock_list = fetcher.get_stock_list()
    trainer_stocks = stock_list['code'].tolist()[:args.stocks]
    fetcher.close()
    
    # 3. 初始化训练器并启动流程
    trainer = MLModelTrainer(db_path=DATABASE_PATH, punish_unbuyable=TrainingConfig.PUNISH_UNBUYABLE)
    
    # 加载数据
    print(f"\n[Step 1] 正在下载/读取历史行情数据...")
    stocks_data = trainer.load_training_data(trainer_stocks, start_date, end_date)
    
    # 准备数据集 (内部会自动触发市场情绪更新)
    print(f"\n[Step 2] 准备特征数据集与标签...")
    X, y, returns, factor_names, dates, unbuyable_mask, limit_groups = trainer.prepare_dataset(
        stocks_data,
        train_start_date=start_date,
        train_end_date=end_date,
        filter_incomplete_cache=not args.force,
        include_fundamentals=True
    )
    
    # 训练模型
    print(f"\n[Step 3] 训练多种机器学习模型并对比评分...")
    results = trainer.train_models(X, y, returns, factor_names, dates, unbuyable_mask=unbuyable_mask, limit_groups=limit_groups)
    
    # 对比与保存
    best_model_type = trainer.compare_models(results)

    if best_model_type is None:
        print("\n[错误] 模型训练全部失败，请检查数据或参数设置。")
        return
    
    # 8. 保存模型
    print("\n保存模型...")
    archive_dir = trainer.save_models(
        save_dir=TrainingConfig.SAVE_DIR, 
        years=TrainingConfig.YEARS_FOR_TRAINING, 
        stocks=len(trainer_stocks)
    )
    
    # 9. 保存因子汇总
    trainer.save_factor_summary(factor_names, save_dir=archive_dir)
    
    
    print(f"\n=== 训练流程全部完成 ===")

if __name__ == "__main__":
    main()
