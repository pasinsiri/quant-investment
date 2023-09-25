import pandas as pd
import yfinance as yf
import json
from functions.indicators import TechnicalIndicators

def flag_new_high_new_low(series, n:int = 20):
    series = series.iloc[series.shape[0] - n:]
    last = series.loc[series.index.max()]
    highest = series.max()
    if last == highest:
        return f'{n} day high'
    lowest = series.min()
    if last == lowest:
        return f'{n} day low'
    else:
        return None

def flag_ma(series, n:int = 20):
    current_index = series.index.max()
    rolling_ma = series.rolling(n).mean()
    if series.loc[current_index] > rolling_ma.loc[current_index]:
        return f'above MA {n}'
    elif series.loc[current_index] < rolling_ma.loc[current_index]:
        return f'below MA {n}'
    else:
        return None

# TODO: technical indicators
ti = TechnicalIndicators()



# TODO: load ticker list
with open('./meta/auto_alert_tickers.json', 'r') as f:
    tickers = json.load(f)

yfinance_meta = yf.Tickers(list(tickers['Indexes'].keys()))
raw_df = yfinance_meta.history(period='1y', auto_adjust=True, progress=False)
raw_df.columns = [(c[0], tickers['Indexes'][c[1]]) for c in raw_df.columns]
raw_df.index = pd.to_datetime(raw_df.index)

close_cols = [c for c in raw_df.columns if c[0] == 'Close']
close_df = raw_df[close_cols]
close_df.columns = [c[1] for c in close_df.columns]
close_df.head()

res = close_df.apply(flag_new_high_new_low, axis=0, n=7)
res = res[res.notnull()]
msg_list = [': '.join([i, res.loc[i]]) for i in res.index]
for msg in msg_list:
    print(msg)
