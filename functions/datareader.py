import pandas_datareader.data as web 
import pandas as pd 
import datetime as dt
import time 
import requests

# TODO: Stock Reader
# ? Yahoo Finance
def pull_stock_data(stocks, start: dt.date, end: dt.date, columns: list = ['Close'], source: str = 'yahoo'):
    """pull stock trading data (prices, volume, etc.)

    Args:
        stocks (str or list): ticker(s) of stocks
        start (dt.date): start date
        end (dt.date): end date
        columns (list, optional): price types (open / high / low / close). Defaults to ['Close'].
        source (str, optional): data source. Defaults to 'yahoo'.

    Returns:
        pandas.DataFrame: a Pandas time-series dataframe contains prices of each stocks of interest
    """
    raw = web.DataReader(stocks, source, start, end)
    cols = [col for col in raw.columns if col[0] in columns]

    df = raw[cols]
    df.columns = ['_'.join(col) for col in df.columns]
    return df

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

    def get_multiple_price_data(self, ticker_list: list, interval: str, start: dt.date, end: dt.date, cols: list = ['close'], clean: bool = True) -> pd.DataFrame:
        """pull cryptocurrency price data from Klines API (Binance or KuCoin)

        Args:
            ticker_list (list): a list of cryptocurrency tickers
            interval (str): price interval for each row (1d / 1h / 1m)
            start (dt.date): start date
            end (dt.date): end date
            cols (list): a price types of interest (open / high / low / close)
            clean (bool, optional): True if we want to clean columns names, otherwise False. Defaults to True.

        Returns:
            pandas.DataFrame: a Pandas' time series dataframe contains prices (open / high / low / close) of all cryptocurrencies
        """
        all_df = pd.DataFrame()
        for t in ticker_list: 
            tmp_df = self.get_price_data(ticker = t, interval = interval, start = start, end = end, clean = clean)
            tmp_df = tmp_df[[cols]].astype(float)
            tmp_df.columns = ['_'.join([t, c]) for c in tmp_df.columns]
            all_df = pd.concat([all_df, tmp_df], axis = 1)
        return all_df