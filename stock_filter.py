import pandas as pd
import datetime as dt
import os
import argparse
from dateutil.relativedelta import relativedelta
from functions.indicators import IndicatorExecutor

# TODO: get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--date', help='Target date', default=None)
parser.add_argument('--market', help='Market (set or mai or all_thai)', default='all_thai')
parser.add_argument('--basepath', help='Data source path')
parser.add_argument('--export', help='Export path')
args = parser.parse_args()

target_date = dt.date.today()
start_date = (target_date - relativedelta(years=1)).replace(day=1)
market = args.market
base_path = os.path.join('data/prices', market)
save = True
export_path = os.path.join('res/thai_stock_filtered', market)
curr_date = dt.date.today()

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

# TODO: Rule 1
"""
1) the close price must be above the 200-day moving average (this indicates a long-term uptrend)
2) the close price must be below the 50-day moving average (this indicates a short-term downtrend)
3) the bollinger_ratio must between -0.05 and 0.1 or 0.45 and 0.6 (this indicates a buy signal)
"""

filtered_df = latest_df[(latest_df['ma_200_pct_deviation'] > 0.0) &
                        (latest_df['ma_50_pct_deviation'] < 0.0) &
                        ((latest_df['bollinger_ratio'].between(-0.05, 0.2)) | \
                            latest_df['bollinger_ratio'].between(0.45, 0.6))]
# filtered_df = latest_df[(latest_df['ma_200_pct_deviation'] > 0.0) &
#                         (latest_df['ma_50_pct_deviation'] < 0.0)]

print(f'{market}: found {len(filtered_df)} tickers')
print(filtered_df.head(30))

if save:
    filtered_df.to_csv(os.path.join(export_path, dt.datetime.strftime(curr_date, '%Y-%m-%d') + '.csv'))