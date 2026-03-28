import sys
import os
import pandas as pd
from datetime import datetime

# 将项目根目录添加到路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.baostock_fetcher import BaostockFetcher
from core.data.baostock_fetcher_methods import fetch_stock_industry


def update_industry_to_db():
    """
    从 Baostock 抓取全市场股票行业信息，并保存到本地数据库
    """
    print("开始更新股票行业分类信息...")
    
    fetcher = BaostockFetcher()
    try:
        # 登录 Baostock
        if not fetcher._bs_login():
            print("Baostock 登录失败，请检查网络和登录凭据")
            return
            
        # 获取行业信息
        # 如果不传 code 和 date，将获取全市场在该日期（通常是今日或最近工作日）的最新分类
        # 默认使用 query_stock_industry(code=None, date=None)
        print("🌐 正在从 Baostock 抓取全市场行业列表...")
        df_industry = fetch_stock_industry()
        
        if df_industry.empty:
            print("⚠ 抓取失败，行业信息为空")
            return
            
        # 保存到数据库
        print(f"成功获取 {len(df_industry)} 条行业记录。正在保存到数据库...")
        fetcher._save_stock_industry_to_db(df_industry)
        
        # 统计行业情况
        industry_counts = df_industry['industry'].value_counts()
        print(f"\n✅ 行业信息同步完成！共有 {len(industry_counts)} 个细分行业")
        print(f"前 5 大行业及涵盖股票数:\n{industry_counts.head(5)}")
        
    except Exception as e:
        print(f"✗ 更新行业数据异常: {e}")
    finally:
        fetcher.close()


if __name__ == "__main__":
    update_industry_to_db()
