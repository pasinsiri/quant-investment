import pandas as pd
import yfinance as yf
import json
import logging
from functions.indicators import TechnicalIndicators


def flag_new_high_new_low(series, n: int = 20):
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


def flag_ma(series, n: int = 20):
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
ticker_values = list(tickers['Indexes'].values())

yfinance_meta = yf.Tickers(ticker_list)
raw_df = yfinance_meta.history(period='1y', auto_adjust=True, progress=False)
raw_df.columns = [(c[0], tickers['Indexes'][c[1]]) for c in raw_df.columns]
raw_df.index = pd.to_datetime(raw_df.index)

# TODO: technical indicators
for ticker in ticker_values:
    print(f'----- {ticker} -----')
    ticker_df = raw_df[[
        c for c in raw_df.columns if c[1] == ticker]].sort_index()
    ticker_df.columns = [c[0].lower() for c in ticker_df.columns]
    ti = TechnicalIndicators(ticker_df)

    # * initialize the terminal dataframe
    res_df = ticker_df.loc[:, ['close']]
    latest_date = res_df.index.max()
    print(f'closes at {res_df.loc[latest_date].values[0]:.2f}')
    logging.info(f'Latest date found is {latest_date}')

    # * add indicators
    res_df['ma_50'] = res_df['close'].rolling(50).mean()
    res_df['ma_50_pct'] = (res_df['close'] - res_df['ma_50']) / res_df['ma_50']
    res_df['lag_ma_50_pct'] = res_df['ma_50_pct'].shift(1)
    res_df['ma_200'] = res_df['close'].rolling(200).mean()
    res_df['ma_200_pct'] = (
        res_df['close'] - res_df['ma_200']) / res_df['ma_200']
    res_df['lag_ma_200_pct'] = res_df['ma_200_pct'].shift(1)
    res_df['cross_signal'] = res_df['ma_50'] - res_df['ma_200']
    res_df['lag_cross_signal'] = res_df['cross_signal'].shift(1)
    res_df['rsi'] = ti.RSI(n=14)
    res_df['macd'], res_df['macd_signal'] = ti.MACD(n_long=26, n_short=12)
    res_df['bollinger_ratio'] = ti.bollinger_ratio(n=20, k=2)

    # * evaluate trigger conditions
    current_data = res_df.loc[latest_date]
    # ? golden cross / death cross
    if current_data.loc['cross_signal'] > 0 and current_data.loc['lag_cross_signal'] < 0:
        print(f'Golden Cross!!!')
    elif current_data.loc['cross_signal'] < 0 and current_data.loc['lag_cross_signal'] > 0:
        print(f'Death Cross!!!')

    # ? cross MA lines (50 / 200) in either direction
    for ma_range in [50, 200]:
        if current_data.loc[f'ma_{ma_range}_pct'] > 0 and current_data.loc[f'lag_ma_{ma_range}_pct'] < 0:
            print(f'Price crosses MA{ma_range} upwards')
        elif current_data.loc[f'ma_{ma_range}_pct'] < 0 and current_data.loc[f'lag_ma_{ma_range}_pct'] > 0:
            print(f'Price crosses MA{ma_range} downwards')

    # ? RSI
    if current_data.loc['rsi'] < 30:
        print(f'RSI is at {current_data.loc["rsi"]:.2f}, which is oversold')
    elif current_data.loc['rsi'] > 70:
        print(f'RSI is at {current_data.loc["rsi"]:.2f}, which is overbought')

    # ? Bollinger Ratio
    if current_data.loc['bollinger_ratio'] > 1 or current_data.loc['bollinger_ratio'] < 0:
        print(
            f'Bollinger ratio is at {current_data.loc["bollinger_ratio"]:.2f}')

    print('\n')
