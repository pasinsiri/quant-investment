import pandas_datareader.data as web 
import pandas as pd 
import datetime as dt

def pull_stock_data(stocks, start, end, price_type = 'Close', source = 'yahoo'):
    raw = web.DataReader(stocks, source, start, end)
    cols = [col for col in raw.columns if col[0] == 'Close']

    df = raw[cols]
    df.columns = [col[1] for col in df.columns]
    return df