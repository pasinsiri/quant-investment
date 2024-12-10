import pandas as pd
import numpy as np
import os

def custom_load(base_path: str, ticker: str, first_year: int, last_year: int):
    paths = []
    for year in range(first_year, last_year + 1):
        for month in range(1, 13):
            m_str = '{:02d}'.format(month)
            path = f'{base_path}/{year}/{m_str}/{ticker}.parquet'
            if os.path.exists(path):
                paths.append(path)
            else:
                # print(f'{path} not found')
                pass
    return paths

def adjust_price(
        df, ticker_list: list, base_path: str, first_year: int, last_year: int, 
        adjust_cols: list = ['Open', 'High', 'Low', 'Close'], split_col_name:str = 'stock split'
):
    for ticker in ticker_list:
        paths = custom_load(base_path, ticker, first_year, last_year)
        ticker_df = pd.read_parquet(*[paths]).sort_index(ascending=False)
        ticker_df['adjust_factor'] = ticker_df[split_col_name] \
                                        .apply(lambda x: 1 if x == 0 else x)
        ticker_df['cum_adj_factor'] = ticker_df['adjust_factor'].cumprod() \
                                        .shift(1).fillna(1)
        for col in adjust_cols:
            ticker_df[col] = ticker_df[col] * ticker_df['cum_adj_factor']
