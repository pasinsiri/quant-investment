import pandas_datareader.data as web 
import pandas as pd 
import datetime as dt
import time 
import requests
import warnings
import os
import yfinance as yf

# TODO: Stock Reader
# ? Yahoo Finance
def pull_stock_data(sectors:dict, start:str, end:str, export_dir:str, source:str = 'yahoo', sleep:int = 2, verbose:bool = False):
    for sector in sectors.keys():
        tickers = sectors[sector]
        parent_dir = f'{export_dir}/{sector}'
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        for t in tickers:
            price_raw = web.DataReader(t + '.BK', start = start, end = end, data_source = source)
            price_raw.insert(0, 'ticker', t)
            parent_subdir = f'{parent_dir}/{t}'
            if not os.path.exists(parent_subdir):
                os.mkdir(parent_subdir)
            for y in range(2015, 2022 + 1):
                year_df = price_raw[price_raw.index.year == y]
                if year_df.shape[0] > 0:
                    export_path = f'{parent_subdir}/{t}_{y}.parquet'
                    year_df.to_parquet(export_path)
                time.sleep(sleep)
            if verbose == True:
                print(f'{t} is completed')
    return

# ? Yahoo Finance with yfinance library
class YFinanceReader():
    def __init__(self, stock_sectors:dict) -> None:
        self.stock_sectors = stock_sectors
        self.ticker_list = [v + '.BK' for s in stock_sectors.values() for v in s]
        self.yfinance_meta = yf.Tickers(self.ticker_list)
        self.is_loaded = False
        pass

    def load_data(self, period:str = 'max'):
        self.price_df = self.yfinance_meta.history(period = period)
        self.is_loaded = True 
        return 
    
    def save(self, parent_dir:str, verbose:bool = False):
        if not self.is_loaded:
            raise ReferenceError('call load_data first before saving')

        for t in self.ticker_list: 
            t_trim = t.replace('.BK', '')
            ticker_dir = f'{parent_dir}/{t_trim}'
            if not os.path.exists(ticker_dir):
                os.mkdir(ticker_dir)

            ticker_cols = [c for c in self.price_df.columns if c[1] == t]
            ticker_df = self.price_df[ticker_cols].dropna(axis = 0)
            ticker_df.columns = [c[0].lower() for c in ticker_df.columns]
            ticker_df.insert(0, 'ticker', t_trim)
            ticker_df.index.name = 'date'

            price_dir = f'{ticker_dir}/price'
            if not os.path.exists(price_dir):
                os.mkdir(price_dir)
            years = sorted(list(set(ticker_df.index.year)))
            for y in years:
                year_df = ticker_df[ticker_df.index.year == y]
                year_df.to_parquet(f'{price_dir}/{str(y)}.parquet')
        if verbose:
            print('saving completed')

# TODO: Cryptocurrency Reader
# ? Klines API (Binance, KuCoin)
class CryptocurrencyReader():
    def __init__(self, source: str) -> None:
        self.source = source.lower()
        self.base_url_mapper = self._list_base_url_mapper()
        if self.source in self.base_url_mapper:
            self.base_url = self.base_url_mapper[self.source]
        else:
            raise ValueError('source does not exist, only support Binance and KuCoin')

    def _list_base_url_mapper(self):
        return {
            'binance': 'https://api.binance.com/api/v3/klines',
            'kucoin': 'https://api.kucoin.com/api/v1/market/candles'
        }

    def get_price_data(self, ticker: str, interval: str, start: dt.date, end: dt.date, clean: bool = True) -> pd.DataFrame:
        """pull cryptocurrency price data from Klines API (Binance or KuCoin)

        Args:
            ticker (str): a cryptocurrency ticker
            interval (str): price interval for each row (1d / 1h / 1m)
            start (dt.date): start date
            end (dt.date): end date
            clean (bool, optional): True if we want to clean columns names, otherwise False. Defaults to True.

        Returns:
            pandas.DataFrame: a Pandas' time series dataframe contains prices (open, high, low, close) of a cryptocurrency of interest
        """

        symbol_url = 'symbol={0}'.format(ticker) 
        param_list = {
            'binance': [1000, 'interval', 'startTime', 'endTime'],
            'kucoin': [1, 'type', 'startAt', 'endAt']
        }
        params = param_list[self.source] 
        start_ts = int(time.mktime(start.timetuple()) * params[0]) 
        end_ts = int(time.mktime(end.timetuple()) * params[0]) 
        interval_url = f'{params[1]}={interval}' 
        start_url = f'{params[2]}={start_ts}' 
        end_url = f'{params[3]}={end_ts}'
        
        full_url = '&'.join([self.base_url + '?' +  symbol_url, interval_url, start_url, end_url])
        raw = requests.get(full_url).json()
        if clean:
            if self.source == 'binance':
                cols = ['raw_open_time', 'open', 'high', 'low', 'close', 'raw_close_time', 'volume', 'cnt_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore1', 'ignore2']
                df = pd.DataFrame(raw, columns = cols)
                df['open_time'] = df['raw_open_time'].apply(lambda x: dt.date.fromtimestamp(float(x) / 1000))
                df.drop(['raw_open_time', 'raw_close_time', 'ignore1', 'ignore2'], axis = 1, inplace = True)
                df = df.set_index('open_time').sort_index()
            elif self.source == 'kucoin':
                cols = ['raw_open_time', 'open', 'close', 'high', 'low', 'volume', 'turnover']
                df = pd.DataFrame(raw['data'], columns = cols)
                df['open_time'] = df['raw_open_time'].apply(lambda x: dt.date.fromtimestamp(float(x)))
                df = df.set_index('open_time').sort_index()
            else:
                raise ValueError('source does not exist, only support Binance and KuCoin')
        else:
            df = raw
        return df

    def get_multiple_price_data(self, ticker_list: list, interval: str, start: dt.date, end: dt.date, ref_ticker:str = '', cols: list = ['close'], col_suffix: bool = False, clean: bool = True) -> pd.DataFrame:
        """pull cryptocurrency price data from Klines API (Binance or KuCoin)

        Args:
            ticker_list (list): a list of cryptocurrency tickers
            interval (str): price interval for each row (1d / 1h / 1m)
            start (dt.date): start date
            end (dt.date): end date
            ref_ticker (str): a reference ticker, usually one of stablecoins (USDC / USDT / BUSD) to be appended to every argument in ticker_list, if pass default value you need to type the full ticker names, e.g. BTCUSDT, in ticker_list
            cols (list, optional): a price types of interest (open / high / low / close). Defaults to ['close']
            col_suffix (bool, optional): if True, price type will be appended after ticker (if pass multiple price types, force to True)
            clean (bool, optional): True if we want to clean columns names, otherwise False. Defaults to True.

        Returns:
            pandas.DataFrame: a Pandas' time series dataframe contains prices (open / high / low / close) of all cryptocurrencies
        """

        # ? if pass multiple price types, force col_suffix to True
        if len(cols) > 1:
            warnings.warn('if pass multiple price types, force col_suffix to True')
            col_suffix = True

        # ? prepare ticker list
        if ref_ticker != '':
            ticker_list = [ticker + ref_ticker for ticker in ticker_list]

        all_df = pd.DataFrame()
        for t in ticker_list: 
            tmp_df = self.get_price_data(ticker = t, interval = interval, start = start, end = end, clean = clean)
            tmp_df = tmp_df[cols].astype(float)
            if col_suffix:
                tmp_df.columns = ['_'.join([t, c]) for c in tmp_df.columns]
            else:
                tmp_df.columns = [t]
            all_df = pd.concat([all_df, tmp_df], axis = 1)
        return all_df