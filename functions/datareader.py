import pandas_datareader.data as web 
import pandas as pd 
import datetime as dt

def pull_stock_data(stocks, start, end, columns = ['Close'], source = 'yahoo', **kwargs):
    raw = web.DataReader(stocks, source, start, end)
    cols = [col for col in raw.columns if col[0] in columns]

    df = raw[cols]
    df.columns = ['_'.join(col) for col in df.columns]
    return df