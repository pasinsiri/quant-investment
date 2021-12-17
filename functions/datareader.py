import pandas_datareader.data as web 
import pandas as pd 
import datetime as dt
import time 

# TODO: Stock Reader
# ? Yahoo Finance
def pull_stock_data(stocks, start, end, columns = ['Close'], source = 'yahoo'):
    raw = web.DataReader(stocks, source, start, end)
    cols = [col for col in raw.columns if col[0] in columns]

    df = raw[cols]
    df.columns = ['_'.join(col) for col in df.columns]
    return df

# TODO: Cryptocurrency Reader
# ? Binance Klines API
def get_binance_data(ticker, interval, start, end, clean = True):
    start_ts = int(time.mktime(start.timetuple()) * 1000)
    end_ts = int(time.mktime(end.timetuple()) * 1000)
    base_url = 'https://api.binance.com/api/v3/klines'
    symbol_url = 'symbol={0}'.format(ticker) 
    interval_url = 'interval={0}'.format(interval) 
    start_url = 'startTime={0}'.format(start_ts) 
    end_url = 'endTime={0}'.format(end_ts)

    full_url = '&'.join([base_url + '?' +  symbol_url, interval_url, start_url, end_url])

    raw = requests.get(full_url).json()
    if clean:
        cols = ['raw_open_time', 'open', 'high', 'low', 'close', 'raw_close_time', 'volume', 'cnt_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore1', 'ignore2']
        df = pd.DataFrame(raw, columns = cols)

        df['open_time'] = df['raw_open_time'].apply(lambda x: dt.date.fromtimestamp(float(x) / 1000))
        df.drop(['raw_open_time', 'raw_close_time', 'ignore1', 'ignore2'], axis = 1, inplace = True)
        df.set_index('open_time', inplace = True)
    else:
        df = raw
    return df

# get multiple assets' price 
def get_binance_multiple_assets(ticker_list, interval, start, end, cols = ['close']):
    all_df = pd.DataFrame()
    for t in ticker_list: 
        tmp_df = get_binance_data(ticker = t, interval = interval, start = start, end = end, clean = True)
        tmp_df = tmp_df[cols].astype(float)
        tmp_df.columns = ['_'.join([t, c]) for c in tmp_df.columns]
        all_df = pd.concat([all_df, tmp_df], axis = 1)
    return all_df