import pyodbc

conn_str = 'DRIVER={SQL Server};SERVER=10.2.47.124;DATABASE=JYDB;UID=jydb_reader;PWD=Syzx805805#'
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

key_tables = [
    ('LC_InstiArchive', '上市公司基本资料'),
    ('LC_CSIIndustry', '上市公司行业板块'),
    ('LC_STIBDailyQuote', '上市公司股票行情'),
    ('LC_BalanceSheet', '上市公司财务报表-资产负债表'),
    ('LC_IncomeStatement', '上市公司财务报表-利润表'),
    ('LC_CashFlowStatement', '上市公司财务报表-现金流量表'),
    ('LC_MainIndexNew', '上市公司财务指标'),
    ('LC_DerivativeData', '上市公司财务衍生指标')
]

with open('table_columns.txt', 'w', encoding='utf-8') as f:
    for table_name, desc in key_tables:
        f.write(f'{"="*80}\n')
        f.write(f'{desc}\n')
        f.write(f'表名: {table_name}\n')
        f.write(f'{"="*80}\n')
        
        query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """
        cursor.execute(query)
        columns = []
        for row in cursor.fetchall():
            columns.append({
                'column_name': row.column_name,
                'data_type': row.data_type,
                'is_nullable': row.is_nullable
            })
        
        f.write(f'字段数: {len(columns)}\n\n')
        for col in columns:
            nullable = '可空' if col['is_nullable'] == 'YES' else '非空'
            f.write(f"  {col['column_name']} ({col['data_type']}, {nullable})\n")
        f.write('\n')

cursor.close()
conn.close()
print('字段信息已保存到 table_columns.txt')