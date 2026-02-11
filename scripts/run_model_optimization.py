"""
运行模型优化脚本

执行特征选择、特征工程超参数优化、集成学习优化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.factors.train_ml_model import MLModelTrainer
from core.factors.model_optimizer import ModelOptimizer
from core.factors.ml_factor_model import EnsembleFactorModel
from config import DATABASE_PATH, TrainingConfig


def main():
    """主函数"""
    print("=" * 80)
    print("模型优化流程")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载训练好的模型
    print("\n[步骤 1] 加载训练数据和模型")
    print("-" * 80)
    
    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    
    # 设置训练参数
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*TrainingConfig.YEARS_FOR_TRAINING)).strftime('%Y-%m-%d')
    
    # 获取股票列表（示例：使用前100只股票）
    import sqlite3
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT DISTINCT code FROM daily_data 
        WHERE date >= ? 
        ORDER BY code 
        LIMIT {TrainingConfig.STOCK_NUM}
    ''', (start_date,))
    stock_codes = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"股票数量: {len(stock_codes)}")
    print(f"日期范围: {start_date} - {end_date}")
    
    # 加载数据
    stocks_data = trainer.load_training_data(stock_codes, start_date, end_date)
    
    if not stocks_data:
        print("错误: 没有加载到数据")
        return
    
    # 2. 准备训练数据
    print("\n[步骤 2] 准备训练数据")
    print("-" * 80)
    
    X, y, factor_names = trainer.prepare_dataset(
        stocks_data,
        forward_days=5,
        threshold=0.02
    )
    
    print(f"样本数: {len(X)}")
    print(f"特征数: {len(factor_names)}")
    print(f"正样本比例: {y.sum() / len(y):.2%}")
    
    # 3. 训练基础模型
    print("\n[步骤 3] 训练基础模型")
    print("-" * 80)
    
    model_types = ['xgboost', 'lightgbm']
    results = trainer.train_models(X, y, factor_names, model_types=model_types)
    
    if not results:
        print("错误: 模型训练失败")
        return
    
    # 4. 运行优化
    print("\n[步骤 4] 运行模型优化")
    print("-" * 80)
    
    optimizer = ModelOptimizer()
    
    optimization_results = optimizer.run_full_optimization(
        models=trainer.models,
        X=X,
        y=y,
        feature_names=factor_names
    )
    
    # 5. 创建优化后的集成模型
    print("\n[步骤 5] 创建优化后的集成模型")
    print("-" * 80)
    
    selected_features = optimization_results['selected_features']
    ensemble_weights = optimization_results.get('ensemble_weights_grid')
    
    if ensemble_weights:
        # 使用优化后的权重创建集成模型
        model_list = [trainer.models[name] for name in optimization_results['model_names']]
        
        optimized_ensemble = EnsembleFactorModel(
            models=model_list,
            weights=ensemble_weights
        )
        
        # 保存优化后的集成模型
        optimized_ensemble.save_model('models/optimized_ensemble_model.pkl')
        print("优化后的集成模型已保存")
    
    # 6. 保存优化结果
    print("\n[步骤 6] 保存优化结果")
    print("-" * 80)
    
    optimizer.save_results('models/optimization_results.json')
    
    # 保存选择的特征列表
    with open('models/selected_features.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(selected_features))
    print(f"选择的特征已保存到: models/selected_features.txt")
    
    # 7. 生成优化报告
    print("\n[步骤 7] 生成优化报告")
    print("-" * 80)
    
    generate_optimization_report(optimization_results, factor_names)
    
    print("\n" + "=" * 80)
    print("优化完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def generate_optimization_report(results: dict, original_features: list):
    """生成优化报告"""
    
    report = []
    report.append("=" * 80)
    report.append("模型优化报告")
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 特征选择结果
    report.append("## 1. 特征选择结果")
    report.append("-" * 80)
    report.append(f"原始特征数: {len(original_features)}")
    
    selected_features = results.get('selected_features', [])
    report.append(f"选择特征数: {len(selected_features)}")
    report.append(f"特征减少: {len(original_features) - len(selected_features)} 个")
    report.append(f"压缩比例: {len(selected_features) / len(original_features):.1%}")
    report.append("")
    
    report.append("选择的特征:")
    for i, feature in enumerate(selected_features, 1):
        score = results.get('feature_scores', {}).get(feature, 0)
        report.append(f"  {i:2d}. {feature:30s} (得分: {score:.6f})")
    report.append("")
    
    # 集成学习优化结果
    report.append("## 2. 集成学习优化结果")
    report.append("-" * 80)
    
    model_names = results.get('model_names', [])
    weights_grid = results.get('ensemble_weights_grid', [])
    
    if weights_grid:
        report.append("优化后的模型权重 (网格搜索):")
        for name, weight in zip(model_names, weights_grid):
            report.append(f"  • {name:15s}: {weight:.3f} ({weight*100:.1f}%)")
        report.append("")
    
    weights_scipy = results.get('ensemble_weights_scipy', [])
    if weights_scipy:
        report.append("优化后的模型权重 (scipy优化):")
        for name, weight in zip(model_names, weights_scipy):
            report.append(f"  • {name:15s}: {weight:.3f} ({weight*100:.1f}%)")
        report.append("")
    
    # Stacking结果
    stacking = results.get('stacking')
    if stacking:
        report.append("Stacking集成结果:")
        report.append(f"  AUC: {stacking['auc']:.4f}")
        report.append("  元模型权重:")
        for name, weight in zip(model_names, stacking['weights']):
            report.append(f"    • {name:15s}: {weight:.3f}")
        report.append("")
    
    # 优化建议
    report.append("## 3. 优化建议")
    report.append("-" * 80)
    
    if len(selected_features) < len(original_features) * 0.5:
        report.append("✓ 特征选择效果显著，成功减少了超过50%的特征")
    else:
        report.append("• 考虑使用更严格的特征选择阈值")
    
    if weights_grid:
        max_weight = max(weights_grid)
        if max_weight > 0.6:
            dominant_model = model_names[weights_grid.index(max_weight)]
            report.append(f"• {dominant_model} 占主导地位 ({max_weight:.1%})，考虑是否需要其他模型")
        else:
            report.append("✓ 模型权重分布较为均衡，集成效果良好")
    
    report.append("")
    report.append("=" * 80)
    
    # 保存报告
    report_text = '\n'.join(report)
    print(report_text)
    
    with open('models/optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n优化报告已保存到: models/optimization_report.txt")


if __name__ == '__main__':
    main()
