import baostock as bs
import pandas as pd

lg = bs.login()
print('login respond error_code:'+lg.error_code)

# Try with adjustfactor
rs_factor = bs.query_history_k_data_plus("sh.600000",
    "date,code,close,adjustfactor",
    start_date='2023-01-01', end_date='2023-01-10',
    frequency="d", adjustflag="3")
print('query_history_k_data_plus with adjustfactor respond error_code:'+rs_factor.error_code)
if rs_factor.error_code == '0':
    data_list_factor = []
    while rs_factor.next():
        data_list_factor.append(rs_factor.get_row_data())
    result_factor = pd.DataFrame(data_list_factor, columns=rs_factor.fields)
    print(result_factor)

bs.logout()
