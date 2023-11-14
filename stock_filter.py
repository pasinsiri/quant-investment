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
    'moving_average_deviation': {'col_name': 'close', 'n': [20, 50, 200]},
    'bollinger_ratio': {'n': 20, 'k': 2},
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
indicator_df = executor.generate_indicator_grid(
    data=raw_df, indicator_params=INDICATOR_PARAMS, ticker_list=None, ticker_col_name='ticker'
)
latest_date = indicator_df.index.max()
latest_df = indicator_df[indicator_df.index == latest_date]
# print(latest_df.tail(20))

# TODO: Rule 1
"""
1) the close price must be above the 200-day moving average (this indicates a long-term uptrend)
2) the close price must be below the 50-day moving average (this indicates a short-term downtrend)
3) the bollinger_ratio must between -0.05 and 0.1 or 0.45 and 0.6 (this indicates a buy signal)
"""
# latest_df['is_long_term_uptrend'] = latest_df['ma_200_pct_deviation'] > 0.0
# latest_df['is_short_term_downtrend'] = latest_df['ma_50_pct_deviation'] < 0.0
# latest_df['is_buy_signal'] = (latest_df['bollinger_ratio'].between(-0.05, 0.1)) | \
#                                 latest_df['bollinger_ratio'].between(-0.45, 0.6)

filtered_df = latest_df[(latest_df['ma_200_pct_deviation'] > 0.0) &
                        (latest_df['ma_50_pct_deviation'] < 0.0) &
                        ((latest_df['bollinger_ratio'].between(-0.05, 0.1)) | \
                            latest_df['bollinger_ratio'].between(-0.45, 0.6))]

print(f'Found {len(filtered_df)} tickers')
print(filtered_df.head(30))