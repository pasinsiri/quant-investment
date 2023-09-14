import pandas_datareader as pdr
import pandas as pd
import numpy as np
import yfinance as yf

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

ticker_list = ['AAPL', 'GOOG', 'NFLX', 'NVDA', 'TSLA']
yfinance_meta = yf.Tickers(ticker_list)
raw_df = yfinance_meta.history(period='1y', auto_adjust=True)
raw_df.index = pd.to_datetime(raw_df.index)

close_cols = [c for c in raw_df.columns if c[0] == 'Close']
close_df = raw_df[close_cols]
close_df.columns = [c[1] for c in close_df.columns]
close_df.head()

res = close_df.apply(flag_new_high_new_low, axis=0, n=7)
res = res[res.notnull()]
res