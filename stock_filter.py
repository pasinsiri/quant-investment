import numpy as np
import pandas as pd
import datetime as dt
import os
from dateutil.relativedelta import relativedelta
from functions.indicators import TechnicalIndicators, IndicatorExecutor

target_date = dt.date.today()
start_date = (target_date - relativedelta(years=1)).replace(day=1)
base_path = './data/prices/set/'

# TODO: set parameters for technical indicators
INDICATOR_PARAMS = {
    'RSI': {'n': 14},
    'bollinger_bands': {'n': 20, 'k': 2, 'concat_result': True},
    'candlestick_volume_ratio': {'mode': 'body'}
}

paths = {}
res = {}
while True:
    y, m = '{:04d}'.format(start_date.year), '{:02d}'.format(start_date.month)
    ym_path = os.path.join(base_path, y, m)
    paths[f'{y}{m}'] = ym_path

    start_date += relativedelta(months=1)
    if start_date > target_date:
        break

res = {key: pd.read_parquet(path) for key, path in paths.items()}
raw_df = pd.concat(res.values(), axis=0)

ticker_list = raw_df['ticker'].unique()
executor = IndicatorExecutor()
indicator_table_list = [
    executor.generate_multiple_indicators(
        TechnicalIndicators(raw_df[raw_df['ticker'] == ticker]), INDICATOR_PARAMS, ticker, True, False
    )
    for ticker in ticker_list
]

indicator_df = pd.concat(indicator_table_list, axis=1)
print(indicator_df.head())