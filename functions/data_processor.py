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

