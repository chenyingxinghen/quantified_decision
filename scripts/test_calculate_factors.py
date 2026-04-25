"""
测试脚本：计算指定股票最新日期的因子数据

功能：
1. 从数据库获取指定股票的最新行情数据
2. 使用 ComprehensiveFactorCalculator 计算所有因子
3. 输出最新日期的因子数据到CSV文件和控制台

使用方法：
    python scripts/test_calculate_factors.py --code 600519 --output factors_output.csv
"""

import sys
import os
import argparse
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
from config import DATABASE_PATH, FactorConfig


def get_latest_stock_data(code: str, db_path: str = DATABASE_PATH, 
                          lookback_days: int = 365) -> Optional[pd.DataFrame]:
    """
    从数据库获取指定股票的最新行情数据
    
    参数:
        code: 股票代码
        db_path: 数据库路径
        lookback_days: 回溯天数（用于计算需要足够长的历史数据）
        
    返回:
        DataFrame: 包含OHLCV等列的股票数据，按日期升序排列
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # 计算起始日期
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # 查询日线数据，关联复权因子实现动态前复权
        query = '''
            SELECT k.code, k.date, k.open, k.high, k.low, k.close, k.preclose, 
                   k.volume, k.amount, k.turnover_rate, k.is_st, 
                   k.peTTM, k.pbMRQ, a.fore_adjust_factor
            FROM daily_data k
            LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
            WHERE k.code = ? AND k.date >= ? AND k.date <= ?
            ORDER BY k.date ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(code, start_date, end_date))
        
        if df.empty:
            print(f"✗ 未找到股票 {code} 的数据")
            return None
        
        print(f"✓ 获取到 {len(df)} 条行情数据 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
        
        # 动态复权处理
        if 'fore_adjust_factor' in df.columns:
            valid_adj = df['fore_adjust_factor'].dropna()
            if not valid_adj.empty:
                base_factor = float(valid_adj.iloc[-1])
                if base_factor != 0:
                    ratio = df['fore_adjust_factor'].ffill().fillna(1.0) / base_factor
                    for col in ['open', 'high', 'low', 'close', 'preclose']:
                        if col in df.columns:
                            df[col] = df[col] * ratio
                    print("✓ 已应用动态前复权")
        
        # 删除复权因子列（后续因子计算不需要）
        if 'fore_adjust_factor' in df.columns:
            df = df.drop(columns=['fore_adjust_factor'])
        
        return df
        
    except Exception as e:
        print(f"✗ 获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        conn.close()


def calculate_factors_for_stock(code: str, data: pd.DataFrame, 
                                output_file: Optional[str] = None,
                                include_fundamentals: bool = True) -> Optional[pd.DataFrame]:
    """
    计算指定股票的所有因子
    
    参数:
        code: 股票代码
        data: 股票行情数据
        output_file: 输出文件路径（可选）
        include_fundamentals: 是否包含基本面因子
        
    返回:
        DataFrame: 包含所有因子的数据
    """
    print(f"\n{'='*60}")
    print(f"开始计算股票 {code} 的因子数据")
    print(f"{'='*60}\n")
    
    try:
        # 初始化因子计算器
        calculator = ComprehensiveFactorCalculator(
            db_path=DATABASE_PATH,
            config=FactorConfig
        )
        
        # 获取目标特征列表（从配置中读取）
        target_features = FactorConfig.FEATURE_COLUMNS if hasattr(FactorConfig, 'FEATURE_COLUMNS') else None
        
        if target_features:
            print(f"目标特征数量: {len(target_features)}")
        else:
            print("使用默认特征集合")
        
        # 计算所有因子
        print("\n正在计算因子...")
        factors_df = calculator.calculate_all_factors(
            code=code,
            data=data,
            apply_feature_engineering=True,
            target_features=target_features,
            verbose=True,
            include_fundamentals=include_fundamentals
        )
        
        if factors_df is None or factors_df.empty:
            print("✗ 因子计算失败，返回空数据")
            return None
        
        print(f"\n✓ 因子计算完成，共 {len(factors_df)} 行，{len(factors_df.columns)} 列")
        
        # 提取最新日期的因子数据
        latest_date = factors_df.index[-1] if hasattr(factors_df.index[-1], 'strftime') else factors_df.index[-1]
        latest_factors = factors_df.iloc[-1:]
        
        print(f"\n最新日期: {latest_date}")
        print(f"最新因子数据形状: {latest_factors.shape}")
        
        # 添加日期和代码列
        latest_factors_with_info = latest_factors.copy()
        latest_factors_with_info.insert(0, 'code', code)
        latest_factors_with_info.insert(1, 'date', latest_date)
        
        # 输出到CSV文件
        if output_file:
            latest_factors_with_info.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n✓ 最新因子数据已保存到: {output_file}")
        
        # 打印部分因子数据预览
        print(f"\n{'='*60}")
        print("最新因子数据预览（前20个因子）:")
        print(f"{'='*60}")
        
        # 选择前20个非零因子展示
        non_zero_cols = [col for col in latest_factors.columns if latest_factors[col].iloc[0] != 0][:20]
        if non_zero_cols:
            preview_df = latest_factors[non_zero_cols].T
            preview_df.columns = ['value']
            preview_df.index.name = 'factor_name'
            print(preview_df.round(4).to_string())
        else:
            print("所有因子值均为0")
        
        # 统计信息
        print(f"\n{'='*60}")
        print("因子数据统计:")
        print(f"{'='*60}")
        print(f"总因子数: {len(factors_df.columns)}")
        print(f"非零因子数: {sum((latest_factors != 0).any())}")
        print(f"零值因子数: {sum((latest_factors == 0).all())}")
        
        # 检查NaN和Inf
        nan_count = latest_factors.isna().sum().sum()
        inf_count = np.isinf(latest_factors.select_dtypes(include=[float])).sum().sum()
        print(f"NaN值数量: {nan_count}")
        print(f"Inf值数量: {inf_count}")
        
        return factors_df
        
    except Exception as e:
        print(f"\n✗ 因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='计算指定股票最新日期的因子数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 计算贵州茅台的因子数据
  python scripts/test_calculate_factors.py --code 600519
  
  # 计算并保存到指定文件
  python scripts/test_calculate_factors.py --code 000001 --output factors_000001.csv
  
  # 不包含基本面因子（更快）
  python scripts/test_calculate_factors.py --code 600519 --no-fundamentals
  
  # 使用更长的历史数据（2年）
  python scripts/test_calculate_factors.py --code 600519 --lookback 730
        """
    )
    
    parser.add_argument('--code', type=str, required=True, 
                       help='股票代码（如：600519, 000001）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出CSV文件路径（可选）')
    parser.add_argument('--db-path', type=str, default=DATABASE_PATH,
                       help=f'数据库路径（默认：{DATABASE_PATH}）')
    parser.add_argument('--lookback', type=int, default=365,
                       help='回溯天数，用于获取足够的历史数据计算因子（默认：365）')
    parser.add_argument('--no-fundamentals', action='store_true',
                       help='不包含基本面因子（可加快计算速度）')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("股票因子数据计算工具")
    print(f"{'='*60}")
    print(f"股票代码: {args.code}")
    print(f"数据库路径: {args.db_path}")
    print(f"回溯天数: {args.lookback}")
    print(f"包含基本面因子: {'否' if args.no_fundamentals else '是'}")
    if args.output:
        print(f"输出文件: {args.output}")
    print(f"{'='*60}\n")
    
    # 步骤1: 获取股票数据
    print("步骤1: 从数据库获取股票行情数据...")
    stock_data = get_latest_stock_data(
        code=args.code,
        db_path=args.db_path,
        lookback_days=args.lookback
    )
    
    if stock_data is None or stock_data.empty:
        print("\n✗ 无法获取股票数据，程序退出")
        sys.exit(1)
    
    # 步骤2: 计算因子
    print("\n步骤2: 计算因子数据...")
    factors_df = calculate_factors_for_stock(
        code=args.code,
        data=stock_data,
        output_file=args.output,
        include_fundamentals=not args.no_fundamentals
    )
    
    if factors_df is None or factors_df.empty:
        print("\n✗ 因子计算失败，程序退出")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✓ 因子计算完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
