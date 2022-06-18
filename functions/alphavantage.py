import requests 
import pandas as pd 
import csv 

class AlphaVantageReader():
    """load company information (overview / cash flow / estimated earnings, etc.) from AlphaVantage using its API
    """
    def __init__(self, key) -> None:
        self.key = key
        self.url_template = f'https://www.alphavantage.co/query?&apikey={self.key}&'

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
        return df

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