import pandas_datareader.data as web 
import pandas as pd 
import numpy as np
import datetime as dt
import time 
import requests
import warnings
import os
import yfinance as yf
import csv 

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
    def __init__(self, stock_sectors:dict, market_suffix:str) -> None:
        self.stock_sectors = stock_sectors
        self.ticker_list = [v + market_suffix for s in stock_sectors.values() for v in s]
        self.yfinance_meta = yf.Tickers(self.ticker_list)
        self.is_loaded = False
        pass

    def _create_month(self, index) -> list:
        ym_arr = pd.DataFrame({'year': index.year, 'month': index.month}).drop_duplicates().to_numpy()
        distinct_list = [''.join(tuple(['{:04d}'.format(row[0]), '{:02d}'.format(row[1])])) for row in ym_arr]
        return distinct_list

    def load_data(self, period:str = 'max'):
        self.price_df = self.yfinance_meta.history(period = period)
        self.is_loaded = True 
        return 
    
    def save(self, parent_dir:str, start_writing_date:dt.date = None, verbose:bool = False):
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

            # * if start_writing_date is defined, filter only data of which date is start_writing_date or later
            if start_writing_date:
                ticker_df = ticker_df[ticker_df.index >= start_writing_date]

            price_dir = f'{ticker_dir}/price'
            if not os.path.exists(price_dir):
                os.mkdir(price_dir)

            # * create a tuple of year and month from the data index
            ym_arr = pd.DataFrame({'year': ticker_df.index.year, 'month': ticker_df.index.month}).drop_duplicates().to_numpy()
            distinct_list = [''.join(tuple(['{:04d}'.format(row[0]), '{:02d}'.format(row[1])])) for row in ym_arr]
            distinct_arr = np.array(distinct_list).reshape(-1, 1)
            months_arr = np.concatenate((ym_arr, distinct_arr), axis = 1)

            for year, month, partition_str in months_arr:
                month_df = ticker_df[
                    (ticker_df.index.year == year) &
                    (ticker_df.index.month == month)
                ]
                month_df.to_parquet(f'{price_dir}/{partition_str}.parquet')

        if verbose:
            print('saving completed')

# ? AlphaVantage API
class AlphaVantageReader():
    """load company information (overview / cash flow / estimated earnings, etc.) from AlphaVantage using its API
    """
    def __init__(self, key) -> None:
        self.key = key
        self.url_template = f'https://www.alphavantage.co/query?&apikey={self.key}&'

    # ? helper functions
    def _load_and_decode(self, url:str) -> pd.DataFrame:
        """(helper) load CSV tables from AlphaVantage, decode and convert to Pandas' DataFrame

        Args:
            url (str): a full endpoint url

        Returns:
            pd.DataFrame: a cleaned DataFrame of data fetched from the API
        """
        with requests.Session() as s:
            download = s.get(url)
            decoded_content = download.content.decode('utf-8')
            cr = list(csv.reader(decoded_content.splitlines(), delimiter=','))
            df = pd.DataFrame(cr[1:], columns = cr[0])
        return df

    def _try_float(self, v:str):
        """try convert obj or str to float

        Args:
            v (str): a value to be converted

        Returns:
            float: a converted value in float format
        """
        if v is None or v == 'None':
            return np.nan 
        else:
            return float(v)

    def _convert_float(self, raw: pd.DataFrame, excepted_keywords:list = ['date']):
        """convert specific columns in a dataframe to float

        Args:
            raw (pd.DataFrame): a DataFrame
            excepted_keywords (list, optional): a list of columns to be ignored in the conversion process (e.g. date columns). Defaults to ['date'].

        Returns:
            pd.DataFrame: a float-converted dataframe
        """
        df = raw.copy()
        for c in df.columns:
            for k in excepted_keywords:
                if k in c.lower():
                    pass 
                else:
                    df[c] = df[c].apply(self._try_float)
        return df

    # * Economic Data
    def get_fed_funds_rate(self, mode:str = 'daily'):
        """get FED's funds rate

        Args:
            mode (str, optional): period of interest (e.g. daily, weekly). Defaults to 'daily'.

        Returns:
            pd.DataFrame: a table of FED's funds rate with date as index
        """
        child_url = f'function=FEDERAL_FUNDS_RATE&interval={mode}'
        r = requests.get(self.url_template + child_url)
        data = r.json()
        df = pd.DataFrame(data['data']).set_index('date').sort_index()
        df['value'] = df['value'].astype(float)
        return self._convert_float(df)

    # * Stock Fundamental Data
    def get_company_data(self, function:str, ticker:str, mode:str = 'quarterly') -> pd.DataFrame:
        """get a company report with a specific period

        Args:
            function (str): report name, can be balance_sheet, income_statement, or cash_flow
            symbol (str): stock ticker
            mode (str, optional): period of interest, can be quarterly or annual. Defaults to 'quarterly'.

        Returns:
            pd.DataFrame: a dataframe contains data of the selected company
        """
        child_url = f'function={function.upper()}&symbol={ticker}'
        r = requests.get(self.url_template + child_url)
        data = r.json() 
        raw_df = data[mode + 'Reports']
        df = pd.concat([pd.Series(x) for x in raw_df], axis = 1).T.set_index('fiscalDateEnding')
        df.insert(0, 'ticker', [ticker] * df.shape[0])
        return df 

    def get_ticker_overview(self, ticker:str) -> pd.Series:
        """get overview information about a specific company

        Args:
            ticker (str): stock ticker

        Returns:
            pd.Series: a series contains overview information of the selected company
        """
        child_url = f'function=OVERVIEW&symbol={ticker}'
        r = requests.get(self.url_template + child_url)
        data = r.json() 
        return pd.Series(data)

    def get_listed_companies(self, is_active:bool = False) -> pd.DataFrame:
        """get all listed tickers

        Args:
            is_active (bool, optional): if true, only active tickers will be fetched, otherwise all tickers. Defaults to False.

        Returns:
            pd.DataFrame: a dataframe of every tickers (companies and ETFs) with its respective ticker, asset type, market, IPO date, delisting date (if any), and status flag
        """
        child_url = 'function=LISTING_STATUS'
        df = self._load_and_decode(self.url_template + child_url)
        if is_active:
            return df[df['status'] == 'active'].reset_index(drop = True) 
        else:
            return df 

    def get_earnings_calendar(self, ticker:str = None, horizon:str = '3month') -> pd.DataFrame:
        """get earnings calendar of every company

        Args:
            ticker (str, optional): if specified, only estimate earnings of the ticker will be selected, if not, estimate earnings of every company will be selected
            horizon (str, optional): future timespan of interest, can be 3month, 6month, or 12month. Defaults to '3month'.

        Returns:
            pd.DataFrame: a dataframe contains estimate earnings of every company
        """
        child_url = f'function=EARNINGS_CALENDAR&horizon={horizon}'
        df = self._load_and_decode(self.url_template + child_url)
        if ticker is None:
            return df
        else:
            selected = df[df['symbol'] == ticker]
            if selected.shape[0] == 0:
                raise ValueError('Ticker does not exist')
            else:
                return selected 

    def get_earnings(self, ticker:str, mode:str = 'quarterly'):
        """get historical earnings with surprises

        Args:
            ticker (str): a stock ticker
            mode (str, optional): period of interests, can be either quarterly or annual. Defaults to 'quarterly'.

        Raises:
            ValueError: mode is not defined

        Returns:
            pd.DataFrame: a dataframe contains earnings and surprises
        """
        child_url = f'function=EARNINGS&symbol={ticker}'
        r = requests.get(self.url_template + child_url)
        data = r.json()
        if mode in ['quarterly', 'annual']:
            raw_array = data[f'{mode}Earnings']
        else:
            raise ValueError('mode not found, can be either quarterly or annual')

        df = pd.concat([pd.Series(q) for q in raw_array], axis = 1).T.set_index('fiscalDateEnding')
        return self._convert_float(df)


    # * Technical data
    def get_moving_average(self, ma_mode:str, ticker:str, time_period:int, interval:str = 'daily', series_type:str = 'close', datatype:str = 'json', add_ticker_to_column_names = False) -> pd.DataFrame:
        """get a Moving Average indicator of a given stock ticker

        Args:
            ma_mode (str): a moving average indicator, can be SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3
            ticker (str): a stock ticker
            time_period (int): a period of interest
            interval (str, optional): an interval of interest, can be daily, weekly, monthly, 1min, 5min, 15min, 30min, 60min. Defaults to 'daily'.
            series_type (str, optional): price types (open / high / low / close). Defaults to 'close'.
            datatype (str, optional): a preferred result type (json / csv). Defaults to 'json'.
            add_ticker_to_column_names (bool, optional): if True, the ticker will be added to column names along with indicator names. Defaults to False.

        Returns:
            pd.DataFrame: a DataFrame contains series of moving average values
        """
        child_url = f'function={ma_mode}&symbol={ticker}&interval={str(interval)}&time_period={time_period}&series_type={series_type}&datatype={datatype}'
        r = requests.get(self.url_template + child_url)
        raw = r.json()

        # ? the request (if JSON) returns a list of two elements, the first one is metadata, and the other is the time series data
        res = pd.DataFrame(list(raw.values())[-1]).T
        if add_ticker_to_column_names:  
            res.columns = ['_'.join([ticker, c]) for c in res.columns]
        return res 

    def get_multiple_moving_average(self, ma_mode_list:str, ticker:str, time_period:int, interval:str = 'daily', series_type:str = 'close', datatype:str = 'json', add_ticker_to_column_names = False) -> pd.DataFrame:
        """get Moving Average indicators using get_moving_average

        Args:
            ma_mode (str): a moving average indicator, can be SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3
            ticker (str): a stock ticker
            time_period (int): a period of interest
            interval (str, optional): an interval of interest, can be daily, weekly, monthly, 1min, 5min, 15min, 30min, 60min. Defaults to 'daily'.
            series_type (str, optional): price types (open / high / low / close). Defaults to 'close'.
            datatype (str, optional): a preferred result type (json / csv). Defaults to 'json'.
            add_ticker_to_column_names (bool, optional): if True, the ticker will be added to column names along with indicator names. Defaults to False.

        Returns:
            pd.DataFrame: a DataFrame contains series of moving average values
        """
        ma_df_list = []
        for ma_mode in ma_mode_list:
            ma_df = self.get_moving_average(ma_mode, ticker, time_period, interval, series_type, datatype, add_ticker_to_column_names)
            ma_df_list.append(ma_df) 
        
        all_ma_df = pd.concat(ma_df_list, axis = 1) 
        return all_ma_df

    def get_sto_rsi(self, ticker:str, interval:str, time_period:int, series_type: str = 'close', fastkperiod: int = 5, fastdperiod: int = 3, fastdmatype:str = 'SMA', datatype:str = 'json', add_ticker_to_column_names = False):
        ma_type_mapper = {
            'SMA': '0',
            'EMA': '1',
            'WMA': '2',
            'DEMA': '3',
            'TEMA': '4',
            'TRIMA': '5',
            'T3': '6',
            'KAMA': '7',
            'MAMA': '8'
        }
        if fastdmatype not in ma_type_mapper.keys():
            raise ValueError('unknown Moving Average type')
        fastdmanum = ma_type_mapper[fastdmatype] 
        child_url = f'function=STOCHRSI&symbol={ticker}&interval={interval}&time_period={str(time_period)}&series_type={series_type}&fastkperiod={str(fastkperiod)}&fastdperiod={str(fastdperiod)}&fastdmatype={fastdmanum}&datatype={datatype}'
        r = requests.get(self.url_template + child_url)
        raw = r.json()
        # ? the request (if JSON) returns a list of two elements, the first one is metadata, and the other is the time series data
        res = pd.DataFrame(list(raw.values())[-1]).T
        if add_ticker_to_column_names:  
            res.columns = ['_'.join([ticker, c]) for c in res.columns]
        return res 

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