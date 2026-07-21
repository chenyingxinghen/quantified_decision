import pyodbc
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import argparse

DATABASE_CONFIGS = {
    "聚源数据库": {
        "server": "10.2.47.124",
        "database": "jydb",
        "username": "jydb_reader",
        "password": "Syzx805805#",
        "driver": "{SQL Server Native Client 11.0}"
    },
    "天相数据库": {
        "server": "10.2.47.80",
        "database": "txdbkq",
        "username": "txdbkq",
        "password": "Syzxtx2304kq!$",
        "driver": "{SQL Server Native Client 11.0}"
    }
}

ALTERNATE_DRIVERS = [
    "{SQL Server}",
    "{ODBC Driver 17 for SQL Server}",
    "{ODBC Driver 13 for SQL Server}",
    "{ODBC Driver 11 for SQL Server}",
    "{SQL Server Native Client 11.0}"
]

TABLE_CATEGORIES = {
    "上市公司基本资料": [
        "LC_InstiArchive",
        "LC_SecuChange",
        "LC_CodeRelationship",
        "LC_ListStatus",
        "LC_CodeChange",
        "LC_NameChange",
        "LC_OrganizationInfo"
    ],
    "上市公司行业板块": [
        "LC_CSIIndustry",
        "LC_ExgIndustry",
        "LC_CorrIndexIndustry",
        "LC_CSIIndusPE",
        "LC_SSIIndusPE",
        "LC_COConcept",
        "LC_ConceptList"
    ],
    "上市公司股票行情": [
        "LC_STIBDailyQuote",
        "LC_STIBAfterDailyQuote",
        "LC_STIBAdjustingFactor",
        "QT_DailyQuote"
    ],
    "上市公司财务报表": [
        "LC_BalanceSheet",
        "LC_BalanceSheetAll",
        "LC_BalanceSheetNew",
        "LC_FBalanceSheet",
        "LC_FBalanceSheetNew",
        "LC_IncomeStatement",
        "LC_IncomeStatementAll",
        "LC_IncomeStatementNew",
        "LC_FIncomeStatement",
        "LC_FIncomeStatementNew",
        "LC_CashFlowStatement",
        "LC_CashFlowStatementAll",
        "LC_CashFlowStatementNew",
        "LC_FCashFlowStatement",
        "LC_FCashFlowStatementNew",
        "LC_QIncomeStatement",
        "LC_QIncomeStatementNew",
        "LC_QCashFlowStatement",
        "LC_QCashFlowStatementNew"
    ],
    "上市公司财务报表附注": [
        "LC_BalanceSheetPS",
        "LC_BalanceSheetPSCN",
        "LC_IncomeStatementPS",
        "LC_IncomeStatementPSCN"
    ],
    "上市公司财务指标": [
        "LC_MainIndexNew",
        "LC_MainDataNew",
        "LC_NewestFinaIndex",
        "LC_MainQuarterData",
        "LC_PerformanceForecast",
        "LC_PerformanceLetters",
        "LC_AuditOpinion",
        "LC_NonRecurringEvent"
    ],
    "上市公司财务衍生指标": [
        "LC_DerivativeData",
        "LC_FSDerivedData",
        "LC_FinanceIndex",
        "LC_FSpecialIndicators",
        "LC_ModelIndex",
        "LC_DIndicesForValuation",
        "LC_IndicesForValuation"
    ]
}

class DatabaseAnalyzer:
    def __init__(self, config_name: str):
        self.config = DATABASE_CONFIGS[config_name]
        self.connection = None
        self.cursor = None

    def connect(self, test_mode=False):
        for driver in ALTERNATE_DRIVERS:
            try:
                conn_str = (
                    f"DRIVER={driver};"
                    f"SERVER={self.config['server']};"
                    f"DATABASE={self.config['database']};"
                    f"UID={self.config['username']};"
                    f"PWD={self.config['password']}"
                )
                self.connection = pyodbc.connect(conn_str, timeout=30)
                self.cursor = self.connection.cursor()
                print(f"成功连接到 {self.config['server']} 数据库 (驱动: {driver})")
                return True
            except Exception as e:
                if not test_mode:
                    print(f"尝试驱动 {driver} 失败: {str(e)[:50]}...")
                continue
        
        print(f"所有驱动尝试连接 {self.config['server']} 均失败")
        return False

    def connect_with_database_name(self, database_name):
        for driver in ALTERNATE_DRIVERS:
            try:
                conn_str = (
                    f"DRIVER={driver};"
                    f"SERVER={self.config['server']};"
                    f"DATABASE={database_name};"
                    f"UID={self.config['username']};"
                    f"PWD={self.config['password']}"
                )
                self.connection = pyodbc.connect(conn_str, timeout=30)
                self.cursor = self.connection.cursor()
                print(f"成功连接到 {self.config['server']} 数据库 {database_name} (驱动: {driver})")
                return True
            except Exception as e:
                print(f"尝试驱动 {driver} 连接 {database_name} 失败: {str(e)[:50]}...")
                continue
        
        return False

    def get_available_databases(self):
        try:
            conn_str = (
                f"DRIVER={ALTERNATE_DRIVERS[0]};"
                f"SERVER={self.config['server']};"
                f"UID={self.config['username']};"
                f"PWD={self.config['password']}"
            )
            self.connection = pyodbc.connect(conn_str, timeout=30)
            self.cursor = self.connection.cursor()
            self.cursor.execute("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb')")
            databases = [row.name for row in self.cursor.fetchall()]
            return databases
        except Exception as e:
            print(f"获取数据库列表失败: {str(e)}")
            return []

    def close(self):
        try:
            if self.cursor:
                self.cursor.close()
        except:
            pass
        try:
            if self.connection:
                self.connection.close()
        except:
            pass
        print("连接已关闭")

    def get_table_size(self, table_name: str) -> Dict:
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            self.cursor.execute(query)
            row_count = int(self.cursor.fetchone()[0])
            
            try:
                query_size = f"""
                    SELECT SUM(reserved_page_count) * 8.0 / 1024 AS size_mb
                    FROM sys.dm_db_partition_stats
                    WHERE object_id = OBJECT_ID('{table_name}')
                """
                self.cursor.execute(query_size)
                size_result = self.cursor.fetchone()
                size_mb = round(float(size_result[0]) if size_result[0] else 0, 2)
            except:
                size_mb = -1
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "size_mb": size_mb
            }
        except Exception as e:
            return {"table_name": table_name, "row_count": -1, "size_mb": -1, "error": str(e)[:50]}

    def analyze_category(self, category_name: str, table_names: List[str]) -> Tuple[Dict, List]:
        print(f"\n{'='*60}")
        print(f"正在分析: {category_name}")
        print(f"{'='*60}")
        
        results = []
        total_rows = 0
        total_size_mb = 0
        missing_tables = []
        
        for table_name in table_names:
            size_info = self.get_table_size(table_name)
            if size_info["row_count"] == -1:
                missing_tables.append(table_name)
                print(f"  ❌ {table_name}: 无法获取大小信息")
            else:
                results.append(size_info)
                total_rows += size_info["row_count"]
                total_size_mb += size_info["size_mb"]
                status = "✅" if size_info["row_count"] > 0 else "⚠️"
                print(f"  {status} {table_name}: {size_info['row_count']:,} 行, {size_info['size_mb']:.2f} MB")
        
        category_summary = {
            "category_name": category_name,
            "table_count": len(table_names),
            "total_rows": total_rows,
            "total_size_mb": round(total_size_mb, 2),
            "total_size_gb": round(total_size_mb / 1024, 2),
            "missing_tables": missing_tables
        }
        
        print(f"\n  汇总: {len(table_names)} 张表, {total_rows:,} 行, {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
        
        return category_summary, results

    def analyze_all_categories(self) -> Dict:
        all_results = {}
        category_summaries = []
        grand_total_rows = 0
        grand_total_size_mb = 0
        
        for category_name, table_names in TABLE_CATEGORIES.items():
            summary, details = self.analyze_category(category_name, table_names)
            all_results[category_name] = {
                "summary": summary,
                "details": details
            }
            category_summaries.append(summary)
            grand_total_rows += summary["total_rows"]
            grand_total_size_mb += summary["total_size_mb"]
        
        print(f"\n{'='*60}")
        print("全部类别汇总")
        print(f"{'='*60}")
        print(f"总表数: {sum(s['table_count'] for s in category_summaries)}")
        print(f"总行数: {grand_total_rows:,}")
        print(f"总大小: {grand_total_size_mb:.2f} MB ({grand_total_size_mb/1024:.2f} GB)")
        
        return all_results

    def get_table_columns(self, table_name: str) -> List[Dict]:
        try:
            query = f"""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """
            self.cursor.execute(query)
            columns = []
            for row in self.cursor.fetchall():
                columns.append({
                    "column_name": row.column_name,
                    "data_type": row.data_type,
                    "is_nullable": row.is_nullable
                })
            return columns
        except Exception as e:
            print(f"获取表 {table_name} 列信息失败: {str(e)}")
            return []

    def sample_table_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        try:
            query = f"SELECT TOP {limit} * FROM {table_name}"
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            print(f"采样表 {table_name} 数据失败: {str(e)}")
            return pd.DataFrame()

    def search_tables(self, keyword: str) -> List[str]:
        try:
            query = f"""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE' 
                AND TABLE_NAME LIKE '%{keyword}%'
                ORDER BY TABLE_NAME
            """
            self.cursor.execute(query)
            tables = [row.TABLE_NAME for row in self.cursor.fetchall()]
            return tables
        except Exception as e:
            print(f"搜索表失败: {str(e)}")
            return []

    def get_all_tables(self) -> List[str]:
        try:
            query = """
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """
            self.cursor.execute(query)
            tables = [row.TABLE_NAME for row in self.cursor.fetchall()]
            return tables
        except Exception as e:
            print(f"获取所有表失败: {str(e)}")
            return []

def generate_factor_analysis() -> Dict:
    factors = {
        "基本面因子": {
            "估值因子": [
                {"name": "PE", "description": "市盈率", "source_table": "FS_ValuationAnalysis", "calculation": "总市值/净利润"},
                {"name": "PB", "description": "市净率", "source_table": "FS_ValuationAnalysis", "calculation": "总市值/净资产"},
                {"name": "PS", "description": "市销率", "source_table": "FS_ValuationAnalysis", "calculation": "总市值/营业收入"},
                {"name": "PCF", "description": "市现率", "source_table": "FS_ValuationAnalysis", "calculation": "总市值/经营现金流"},
                {"name": "EV_EBITDA", "description": "企业价值倍数", "source_table": "FS_ValuationAnalysis", "calculation": "EV/EBITDA"}
            ],
            "盈利因子": [
                {"name": "ROE", "description": "净资产收益率", "source_table": "FS_LatestFinancialIndicators", "calculation": "净利润/平均净资产"},
                {"name": "ROA", "description": "总资产收益率", "source_table": "FS_LatestFinancialIndicators", "calculation": "净利润/平均总资产"},
                {"name": "ROIC", "description": "投入资本回报率", "source_table": "FS_MainFinancialAnalysis", "calculation": "税后净营业利润/投入资本"},
                {"name": "GrossMargin", "description": "毛利率", "source_table": "FS_MainFinancialAnalysis", "calculation": "毛利/营业收入"},
                {"name": "NetMargin", "description": "净利率", "source_table": "FS_MainFinancialAnalysis", "calculation": "净利润/营业收入"},
                {"name": "OperatingMargin", "description": "营业利润率", "source_table": "FS_MainFinancialAnalysis", "calculation": "营业利润/营业收入"}
            ],
            "成长因子": [
                {"name": "RevenueGrowth", "description": "营收增长率", "source_table": "FS_MainAccountingData", "calculation": "(本期营收-上期营收)/上期营收"},
                {"name": "ProfitGrowth", "description": "净利润增长率", "source_table": "FS_MainAccountingData", "calculation": "(本期净利润-上期净利润)/上期净利润"},
                {"name": "EPSGrowth", "description": "每股收益增长率", "source_table": "FS_LatestFinancialIndicators", "calculation": "(本期EPS-上期EPS)/上期EPS"},
                {"name": "AssetGrowth", "description": "资产增长率", "source_table": "FS_MainAccountingData", "calculation": "(本期资产-上期资产)/上期资产"},
                {"name": "OperatingCashFlowGrowth", "description": "经营现金流增长率", "source_table": "FS_MainAccountingData", "calculation": "(本期经营现金流-上期经营现金流)/上期经营现金流"}
            ],
            "质量因子": [
                {"name": "AssetTurnover", "description": "资产周转率", "source_table": "FS_MainFinancialAnalysis", "calculation": "营业收入/平均总资产"},
                {"name": "CurrentRatio", "description": "流动比率", "source_table": "FS_BalanceSheet", "calculation": "流动资产/流动负债"},
                {"name": "QuickRatio", "description": "速动比率", "source_table": "FS_BalanceSheet", "calculation": "(流动资产-存货)/流动负债"},
                {"name": "DebtToEquity", "description": "资产负债率", "source_table": "FS_BalanceSheet", "calculation": "总负债/总资产"},
                {"name": "CashConversionCycle", "description": "现金转换周期", "source_table": "FS_BalanceSheetNotes", "calculation": "应收账款周转天数+存货周转天数-应付账款周转天数"},
                {"name": "OperatingCashFlowToRevenue", "description": "经营现金流/营收", "source_table": "FS_CashFlowStatement", "calculation": "经营现金流/营业收入"}
            ]
        },
        "市场因子": {
            "规模因子": [
                {"name": "MarketCap", "description": "总市值", "source_table": "QT_DailyQuote", "calculation": "收盘价*总股本"},
                {"name": "CirculatingCap", "description": "流通市值", "source_table": "QT_DailyQuote", "calculation": "收盘价*流通股本"}
            ],
            "动量因子": [
                {"name": "Momentum1M", "description": "1月动量", "source_table": "QT_DailyQuote", "calculation": "过去20日收益率"},
                {"name": "Momentum3M", "description": "3月动量", "source_table": "QT_DailyQuote", "calculation": "过去60日收益率"},
                {"name": "Momentum6M", "description": "6月动量", "source_table": "QT_DailyQuote", "calculation": "过去120日收益率"},
                {"name": "Momentum12M", "description": "12月动量", "source_table": "QT_DailyQuote", "calculation": "过去240日收益率"},
                {"name": "RSI", "description": "相对强弱指数", "source_table": "QT_DailyQuote", "calculation": "14日RSI"}
            ],
            "波动因子": [
                {"name": "Volatility1M", "description": "1月波动率", "source_table": "QT_DailyQuote", "calculation": "过去20日收益率标准差"},
                {"name": "Volatility3M", "description": "3月波动率", "source_table": "QT_DailyQuote", "calculation": "过去60日收益率标准差"},
                {"name": "Beta", "description": "贝塔系数", "source_table": "FS_ModelIndicators", "calculation": "股票与市场的相关性"}
            ],
            "流动性因子": [
                {"name": "Volume", "description": "成交量", "source_table": "QT_DailyQuote", "calculation": "当日成交量"},
                {"name": "Turnover", "description": "换手率", "source_table": "QT_DailyQuote", "calculation": "成交量/流通股本"},
                {"name": "AmihudIlliquidity", "description": "非流动性指标", "source_table": "QT_DailyQuote", "calculation": "|收益率|/成交额"},
                {"name": "FundFlow", "description": "资金流向", "source_table": "QT_FundFlow", "calculation": "净流入金额"}
            ]
        },
        "行业因子": {
            "行业分类": [
                {"name": "IndustryCode", "description": "行业代码", "source_table": "LC_CompanyIndustry", "calculation": "中证行业分类代码"},
                {"name": "IndustryName", "description": "行业名称", "source_table": "LC_CompanyIndustry", "calculation": "行业名称"},
                {"name": "IndustryPE", "description": "行业市盈率", "source_table": "LC_IndustryPE_CSRC", "calculation": "行业平均市盈率"},
                {"name": "ConceptPlate", "description": "概念板块", "source_table": "LC_CompanyConcept", "calculation": "所属概念板块"}
            ]
        },
        "质量因子": {
            "盈利质量": [
                {"name": "Accruals", "description": "应计项目", "source_table": "FS_CashFlowStatement", "calculation": "(净利润-经营现金流)/总资产"},
                {"name": "OperatingAccruals", "description": "经营应计", "source_table": "FS_BalanceSheet", "calculation": "Δ应收账款+Δ存货-Δ应付账款"},
                {"name": "EarningsQuality", "description": "盈利质量", "source_table": "FS_AuditOpinion", "calculation": "审计意见类型"}
            ],
            "治理因子": [
                {"name": "Top1HolderRatio", "description": "第一大股东持股比例", "source_table": "LC_ShareholderStatistics", "calculation": "第一大股东持股数/总股本"},
                {"name": "InstitutionalHoldings", "description": "机构持股比例", "source_table": "LC_ShareholderStatistics", "calculation": "机构持股数/总股本"},
                {"name": "ManagementOwnership", "description": "管理层持股比例", "source_table": "LC_ManagementShareholding", "calculation": "管理层持股数/总股本"}
            ]
        }
    }
    
    return factors

def save_results(results: Dict, factors: Dict, filename: str = "database_analysis_results.json"):
    output = {
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "database_config": list(DATABASE_CONFIGS.keys()),
        "category_analysis": results,
        "factor_analysis": factors
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {filename}")

def main():
    parser = argparse.ArgumentParser(description='恒生聚源金融数据库分析工具')
    parser.add_argument('--database', '-d', choices=['聚源数据库', '天相数据库'], 
                        default='聚源数据库', help='选择要连接的数据库')
    parser.add_argument('--test', '-t', action='store_true', help='测试模式，仅检测可用数据库')
    parser.add_argument('--dbname', '-n', type=str, default=None, help='指定数据库名称')
    parser.add_argument('--search', '-s', type=str, default=None, help='搜索表名')
    parser.add_argument('--list-all', '-l', action='store_true', help='列出所有表')
    args = parser.parse_args()
    
    print("="*60)
    print("恒生聚源金融数据库分析工具")
    print("="*60)
    
    selected_db = args.database
    print(f"\n选择数据库: {selected_db}")
    
    analyzer = DatabaseAnalyzer(selected_db)
    
    if args.test:
        print("\n检测可用数据库...")
        databases = analyzer.get_available_databases()
        if databases:
            print(f"发现 {len(databases)} 个数据库:")
            for db in databases:
                print(f"  - {db}")
        else:
            print("未发现可用数据库")
        return
    
    if args.dbname:
        if not analyzer.connect_with_database_name(args.dbname):
            print(f"连接数据库 {args.dbname} 失败")
            return
    else:
        if not analyzer.connect():
            print("\n尝试获取可用数据库列表...")
            databases = analyzer.get_available_databases()
            if databases:
                print(f"发现以下数据库，请使用 --dbname 参数指定:")
                for db in databases:
                    print(f"  - {db}")
            return
    
    try:
        if args.list_all:
            print("\n列出所有表:")
            tables = analyzer.get_all_tables()
            for i, table in enumerate(tables, 1):
                print(f"  {i}. {table}")
            print(f"\n共 {len(tables)} 张表")
            analyzer.close()
            return
        
        if args.search:
            print(f"\n搜索表名包含 '{args.search}' 的表:")
            tables = analyzer.search_tables(args.search)
            if tables:
                for i, table in enumerate(tables, 1):
                    print(f"  {i}. {table}")
            else:
                print("  未找到匹配的表")
            analyzer.close()
            return
        
        results = analyzer.analyze_all_categories()
        
        print("\n" + "="*60)
        print("生成日截面选股因子分析")
        print("="*60)
        factors = generate_factor_analysis()
        
        print("\n主要因子类别:")
        for category, sub_factors in factors.items():
            print(f"\n  {category}:")
            for sub_category, items in sub_factors.items():
                factor_names = [f['name'] for f in items]
                print(f"    • {sub_category}: {', '.join(factor_names)}")
        
        save_results(results, factors)
        
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()