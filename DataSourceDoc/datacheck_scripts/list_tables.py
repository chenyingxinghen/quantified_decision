import pyodbc

conn_str = 'DRIVER={SQL Server};SERVER=10.2.47.124;DATABASE=JYDB;UID=jydb_reader;PWD=Syzx805805#'
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

query = """
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_TYPE = 'BASE TABLE' 
    AND TABLE_NAME LIKE 'LC[_]%'
    ORDER BY TABLE_NAME
"""
cursor.execute(query)
tables = [row.TABLE_NAME for row in cursor.fetchall()]

with open('lc_tables.txt', 'w', encoding='utf-8') as f:
    f.write(f'LC_开头的表共 {len(tables)} 个:\n')
    for table in tables:
        f.write(f'{table}\n')

print(f'已保存 {len(tables)} 个表名到 lc_tables.txt')

cursor.close()
conn.close()