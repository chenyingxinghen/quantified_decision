"""
因子一致性验证
验证 ComprehensiveFactorCalculator 是否能产生模型所需的完整特征集
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
from core.factors.ml_factor_model import MLFactorModel
from config import DATABASE_PATH

def test_factor_consistency():
    print("\n" + "="*80)
    print("因子一致性验证")
    print("="*80)
    
    # 1. 初始化计算器
    calculator = ComprehensiveFactorCalculator()
    
    # 2. 加载模型
    model_path = 'models/xgboost_factor_model.pkl'
    if not os.path.exists(model_path):
        print(f"  错误: 模型文件不存在: {model_path}")
        return
        
    model = MLFactorModel()
    model.load_model(model_path)
    print(f"  已加载模型: {model_path}")
    print(f"  模型特征数: {len(model.feature_names)}")
    
    # 3. 获取测试数据 (从数据库获取一只股票的最新数据)
    test_code = '000001'
    print(f"\n  获取测试股票数据: {test_code}")
    
    conn = sqlite3.connect(DATABASE_PATH)
    query = f"SELECT * FROM daily_data WHERE code = '{test_code}' ORDER BY date DESC LIMIT 200"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("  错误: 未能获取到股票数据")
        return
        
    df = df.sort_values('date').reset_index(drop=True)
    
    # 4. 计算因子
    print("\n  正在计算综合因子...")
    factors = calculator.calculate_all_factors(
        code=test_code, 
        data=df, 
        apply_feature_engineering=True,
        target_features=model.feature_names  # 传入目标特征
    )
    
    print(f"  计算完成，生成特征数: {len(factors.columns)}")
    
    # 5. 验证缺失特征
    missing_features = [f for f in model.feature_names if f not in factors.columns]
    
    if missing_features:
        print(f"\n  [失败] 仍缺失 {len(missing_features)} 个特征:")
        for f in missing_features[:10]:
            print(f"    - {f}")
        if len(missing_features) > 10:
            print("    ...")
    else:
        print("\n  [成功] 所有特征都已生成，无缺失特征")
        
    # 6. 验证预测
    print("\n  尝试使用生成的因子进行预测...")
    try:
        latest_factors = factors.tail(1)
        prediction = model.predict(latest_factors)
        print(f"  预测成功! 结果: {prediction}")
    except Exception as e:
        print(f"  [失败] 预测出错: {e}")

if __name__ == '__main__':
    test_factor_consistency()
