import pyodbc

conn_str = 'DRIVER={SQL Server};SERVER=10.2.47.124;DATABASE=JYDB;UID=jydb_reader;PWD=Syzx805805#'
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

tables_to_test = [
    'LC_InstiArchive',
    'LC_BalanceSheet',
    'LC_IncomeStatement',
    'LC_CSIIndustry',
    'LC_STIBDailyQuote',
    'LC_MainIndexNew',
    'LC_DerivativeData'
]

for table_name in tables_to_test:
    try:
        query = f"SELECT COUNT(*) FROM {table_name}"
        cursor.execute(query)
        count = cursor.fetchone()[0]
        print(f"{table_name}: {count:,} 行")
    except Exception as e:
        print(f"{table_name}: 错误 - {str(e)[:100]}")

cursor.close()
conn.close()