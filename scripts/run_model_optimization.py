"""
模型优化脚本 - 深度优化版

功能：
1. 智能化数据回填与加载
2. 自动化特征选择 (基于混合重要性评分)
3. 因子超参数搜索 (基于 IC 优化) - 可选
4. 多模型集成权重优化 (网格搜索 & Scipy 优化)
5. 详细的优化审计报告与可视化建议
"""

import sys
import os
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Any, Optional

# 设置项目根目录
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from core.factors.train_ml_model import MLModelTrainer
from core.factors.model_optimizer import ModelOptimizer
from core.factors.ml_factor_model import EnsembleFactorModel
from config import DATABASE_PATH, TrainingConfig, FactorConfig, OptimizationConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "backtest_result" / "optimization.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """确保必要的目录存在"""
    dirs = ['models', 'database/system_data/factors_cache', 'backtest_result']
    for d in dirs:
        (project_root / d).mkdir(parents=True, exist_ok=True)

def get_stock_codes(db_path: str, limit: int, start_date: str) -> List[str]:
    """从数据库获取活跃股票列表"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 选取最近一年有交易记录的股票，按代码排序
        cursor.execute(f'''
            SELECT DISTINCT code FROM daily_data 
            WHERE date >= ? 
            ORDER BY code 
            LIMIT ?
        ''', (start_date, limit))
        codes = [row[0] for row in cursor.fetchall()]
        conn.close()
        return codes
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return []

def main():
    """主逻辑"""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("🚀 启动自动化模型优化流程")
    logger.info("=" * 80)
    
    setup_directories()
    
    # ---------------------------------------------------------
    # 步骤 1: 数据准备
    # ---------------------------------------------------------
    logger.info("\n[步骤 1/6] 加载并预处理训练数据")
    
    trainer = MLModelTrainer(db_path=DATABASE_PATH, punish_unbuyable=TrainingConfig.PUNISH_UNBUYABLE)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_TRAINING)).strftime('%Y-%m-%d')
    
    stock_codes = get_stock_codes(DATABASE_PATH, TrainingConfig.STOCK_NUM, start_date)
    if not stock_codes:
        logger.error("未能找到符合条件的股票，流程终止")
        return

    logger.info(f"股票池规模: {len(stock_codes)} 只")
    logger.info(f"历史回溯期: {start_date} 至 {end_date} (约 {TrainingConfig.YEARS_FOR_TRAINING} 年)")
    
    # 加载原始行情 (带 ST 过滤)
    stocks_data = trainer.load_training_data(stock_codes, start_date, end_date)
    if not stocks_data:
        logger.error("数据加载失败，请检查数据库连接及 daily_data 表内容")
        return
    
    # 计算并准备特征集 (含缓存逻辑)
    # 使用所有 CPU 核心并行加速
    X, y, returns, factor_names, dates, unbuyable, limit_groups = trainer.prepare_dataset(
        stocks_data,
        n_jobs=-1,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS
    )
    
    logger.info(f"特征矩阵规模: {X.shape[0]} 样本 x {X.shape[1]} 特征")
    logger.info(f"正样本率: {np.mean(y >= 0.5):.2%} (软标签均值: {np.mean(y):.4f})")
    
    # ---------------------------------------------------------
    # 步骤 2: 基础模型训练
    # ---------------------------------------------------------
    logger.info("\n[步骤 2/6] 训练全量特征基础模型")
    
    # 评估默认模型，通常包括 XGBoost 和 LightGBM
    model_types = ['xgboost', 'lightgbm']
    train_results = trainer.train_models(
        X, y, returns, factor_names, dates, unbuyable, limit_groups, 
        model_types=model_types
    )
    
    if not train_results:
        logger.error("基础模型训练异常，流程终止")
        return

    # ---------------------------------------------------------
    # 步骤 3: 运行深度优化 (特征选择 + 权重搜索)
    # ---------------------------------------------------------
    logger.info("\n[步骤 3/6] 执行深度优化流程 (特征压缩 + 集成调优)")
    
    optimizer = ModelOptimizer()
    
    # 运行全流程优化
    # 注意: 如果需要优化因子计算周期 (Step 2)，请在 config 中开启 OPTIMIZE_FACTOR_PERIODS
    # 此处我们重点关注特征选择和集成权重
    optimization_results = optimizer.run_full_optimization(
        models=trainer.models,
        X=X,
        y=y,
        feature_names=factor_names
    )
    
    # ---------------------------------------------------------
    # 步骤 4: 构建与保存优化结果
    # ---------------------------------------------------------
    logger.info("\n[步骤 4/6] 保存优化模型与配置")
    
    selected_features = optimization_results.get('selected_features', factor_names)
    ensemble_weights = optimization_results.get('ensemble_weights_grid')
    model_names = optimization_results.get('model_names', model_types)
    
    # 保存特征列表（供后续实盘/回测使用）
    feature_file = project_root / 'models' / 'selected_features.txt'
    with open(feature_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(selected_features))
    logger.info(f"✓ 特征清单已持久化: {feature_file} (数量: {len(selected_features)})")
    
    # 构建最终集成模型
    if ensemble_weights:
        model_list = [trainer.models[name] for name in model_names]
        optimized_ensemble = EnsembleFactorModel(models=model_list, weights=ensemble_weights)
        
        ensemble_path = project_root / 'models' / 'optimized_ensemble_model.pkl'
        optimized_ensemble.save_model(str(ensemble_path))
        logger.info(f"✓ 集成模型已保存: {ensemble_path}")

    # 保存元数据结果
    results_json = project_root / 'models' / 'optimization_results.json'
    optimizer.save_results(str(results_json))
    
    # ---------------------------------------------------------
    # 步骤 5: 生成分析报告
    # ---------------------------------------------------------
    logger.info("\n[步骤 5/6] 生成最终审计报告")
    generate_comprehensive_report(optimization_results, factor_names, train_results)
    
    # ---------------------------------------------------------
    # 步骤 6: 流程总结
    # ---------------------------------------------------------
    logger.info("\n" + "=" * 80)
    duration = datetime.now() - start_time
    logger.info(f"✅ 优化流程圆满完成!")
    logger.info(f"⏱️ 总耗时: {duration.total_seconds() / 60:.1f} 分钟")
    logger.info(f"📂 产出物存于: {project_root / 'models'}")
    logger.info("=" * 80)

def generate_comprehensive_report(results: dict, original_features: list, train_results: dict):
    """生成详细的优化审计报告"""
    
    report = []
    report.append("=" * 80)
    report.append(f"量化策略模型优化报告 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    report.append("=" * 80)
    report.append("")
    
    # 1. 基础模型性能
    report.append("📊 1. 基础模型性能概览")
    report.append("-" * 40)
    report.append(f"{'模型':<15} | {'训练集 AUC':<12} | {'验证集 AUC':<12} | {'Rank IC':<10}")
    report.append("-" * 60)
    for m_name, m_res in train_results.items():
        train_auc = m_res.get('train_metrics', {}).get('auc', 0)
        val_auc = m_res.get('val_metrics', {}).get('auc', 0)
        rank_ic = m_res.get('val_metrics', {}).get('rank_ic', 0)
        report.append(f"{m_name.upper():<15} | {train_auc:12.4f} | {val_auc:12.4f} | {rank_ic:10.4f}")
    report.append("")
    
    # 2. 特征选择分析
    report.append("🔍 2. 特征工程审计")
    report.append("-" * 40)
    selected = results.get('selected_features', [])
    reduction = len(original_features) - len(selected)
    report.append(f"• 原始特征维度: {len(original_features)}")
    report.append(f"• 选定特征维度: {len(selected)}")
    report.append(f"• 降维比例: {reduction / len(original_features):.1%} (减少 {reduction} 个)")
    report.append("")
    
    report.append("核心因子重要性 TOP 10:")
    scores = results.get('feature_scores', {})
    top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (name, val) in enumerate(top_features, 1):
        report.append(f"  {i:2d}. {name:<30} | 评分: {abs(val):.6f}")
    report.append("")
    
    # 3. 集成权重优化
    report.append("⚖️ 3. 集成权重配置")
    report.append("-" * 40)
    model_names = results.get('model_names', [])
    weights_grid = results.get('ensemble_weights_grid', [])
    
    if weights_grid:
        report.append("最优加权方案 (网格搜索):")
        for name, w in zip(model_names, weights_grid):
            report.append(f"  • {name.upper():<15} : {w:.1%} {'(主导)' if w > 0.5 else ''}")
    
    stacking = results.get('stacking')
    if stacking:
        report.append(f"Stacking 元模型 AUC: {stacking.get('auc', 0):.4f}")
    report.append("")
    
    # 4. 行动建议
    report.append("💡 4. 策略行动建议")
    report.append("-" * 40)
    
    # 基于过拟合程度的建议
    high_gap = False
    for m_name, m_res in train_results.items():
        gap = m_res.get('train_metrics', {}).get('auc', 0) - m_res.get('val_metrics', {}).get('auc', 0)
        if gap > 0.1:
            report.append(f"⚠️ 警告: {m_name} 存在明显过拟合 (训练/验证差距 {gap:.2f})，建议增加正则化强度")
            high_gap = True
    
    if not high_gap:
        report.append("✅ 模型泛化能力表现稳健")
        
    if len(selected) > 100:
        report.append("ℹ️ 特征数量较多，可尝试在 config 中调小 N_FEATURES_TO_SELECT 以提高计算速度")
    
    if weights_grid and any(w < 0.05 for w in weights_grid):
        ignored_model = model_names[weights_grid.index(min(weights_grid))]
        report.append(f"ℹ️ 模型 {ignored_model} 贡献度极低，未来可考虑从集成列表中移除")

    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    report_file = project_root / 'models' / 'optimization_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"✓ 深度分析报告已生成: {report_file}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n用户手动终止优化流程")
    except Exception as e:
        logger.error(f"\n❌ 流程执行出错:\n{e}", exc_info=True)
