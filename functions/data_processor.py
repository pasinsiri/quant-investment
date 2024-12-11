import pandas as pd
import numpy as np
import os

def get_parquet_paths(base_path: str, first_year: int, last_year: int, ticker: str = None):
    paths = []
    for year in range(first_year, last_year + 1):
        for month in range(1, 13):
            m_str = '{:02d}'.format(month)
            # ? if no ticker is parsed, all available in a specific year and month will be returned
            ym_path = f'{base_path}/{year}/{m_str}'
            if ticker is None:
                paths.extend(os.listdir(ym_path))
            else:
                path = f'{ym_path}/{ticker}.parquet'
                if os.path.exists(path):
                    paths.append(path)
                else:
                    # print(f'{path} not found')
                    pass
    return paths

def adjust_price(
        df, ticker_list: list, base_path: str, export_base_path: str, first_year: int, last_year: int, 
        dtype_dict: dict, adjust_cols: list = ['open', 'high', 'low', 'close'], split_col_name:str = 'stock splits'
):
    for ticker in ticker_list:
        paths = get_parquet_paths(base_path, ticker, first_year, last_year)

        # * if data for the ticker does not exist, skip it
        if len(paths) == 0:
            continue

        ticker_df = pd.read_parquet(*[paths]).sort_index(ascending=False)
        ticker_df['adjust_factor'] = ticker_df[split_col_name] \
                                        .apply(lambda x: 1 if x == 0 else x)
        ticker_df['cum_adj_factor'] = ticker_df['adjust_factor'].cumprod() \
                                        .shift(1).fillna(1)
        for col in adjust_cols:
            ticker_df[col] = ticker_df[col] * ticker_df['cum_adj_factor']
        ticker_df = ticker_df.sort_index()

        # TODO: cast data type
        for col, dtype in dtype_dict.items():
            ticker_df[col] = ticker_df[col].astype(dtype)
        
        # TODO: save data
        for path in paths:
            path_split = path.split('/')
            year, month = path_split[4], path_split[5]
            month_df = ticker_df[(ticker_df.index.year == int(year)) & 
                                 (ticker_df.index.month == int(month))]
            # export_path = f'{export_base_path}/{year}/{month}/{ticker}.parquet'
            year_path = f'{export_base_path}/{year}'
            if not os.path.exists(year_path):
                os.mkdir(year_path)
            month_path = f'{year_path}/{month}'
            if not os.path.exists(month_path):
                os.mkdir(month_path)
            month_df.to_parquet(f'{month_path}/{ticker}.parquet')