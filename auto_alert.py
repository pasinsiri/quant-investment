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

# TODO: load ticker list
with open('./keys/global_indexes.json', 'r') as f:
    tickers = json.load(f)
ticker_list = list(tickers['Indexes'].keys())

yfinance_meta = yf.Tickers(ticker_list)
raw_df = yfinance_meta.history(period='1y', auto_adjust=True, progress=False)
raw_df.columns = [(c[0], tickers['Indexes'][c[1]]) for c in raw_df.columns]
raw_df.index = pd.to_datetime(raw_df.index)

close_cols = [c for c in raw_df.columns if c[0] == 'Close']
# close_df = raw_df[close_cols]
# close_df.columns = [c[1] for c in close_df.columns]
# close_df.head()

# TODO: technical indicators
for ticker in ticker_list:
    ticker_df = raw_df[raw_df[ticker]].sort_index()
    ticker_df.columns = [c[1].lower() for c in ticker_df.columns]
    ti = TechnicalIndicators(ticker_df)

    # * initialize the terminal dataframe
    res_df = ticker_df[['close']]

    # * add indicators
    res_df['ma_50'] = res_df['close'].rolling(50).mean()
    res_df['ma_50_pct'] = (res_df['close'] - res_df['ma_50']) / res_df['ma_50']
    res_df['lag_ma_50_pct'] = res_df['ma_50_pct'].shift(1)
    res_df['ma_200'] = res_df['close'].rolling(200).mean()
    res_df['ma_200_pct'] = (res_df['close'] - res_df['ma_200']) / res_df['ma_200']
    res_df['lag_ma_200_pct'] = res_df['ma_200_pct'].shift(1)
    res_df['cross_signal'] = res_df['ma_50'] - res_df['ma_200']
    res_df['lag_cross_signal'] = res_df['cross_signal'].shift(1)
    res_df['rsi'] = ti.RSI(n=14)
    res_df['macd'], res_df['macd_signal'] = ti.MACD(n_long=26, n_short=12)
    res_df['bollinger_ratio'] = ti.bollinger_ratio(n=20, k=2)

    # evaluate trigger conditions
    
    

# res = close_df.apply(flag_new_high_new_low, axis=0, n=7)
# res = res[res.notnull()]
# msg_list = [': '.join([i, res.loc[i]]) for i in res.index]
# for msg in msg_list:
#     print(msg)
