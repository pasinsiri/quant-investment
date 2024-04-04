import pandas as pd
import numpy as np


class TechnicalIndicators():
    def __init__(self, ohlcv_df):
        self.ohlcv_df = ohlcv_df

    def parse_cols(self, col_list: list):
        """Parse the columns from the raw dataset to the final dataset.
        For example, if you want close price in the final dataset, use this function

        Args:
            col_list (list): a list of required columns in the OHLCV dataset

        Returns:
            pd.DataFrame: a subset of the OHLCV dataframe with the selected columns
        """
        return self.ohlcv_df[col_list]

    def _get_min_max(self, col_name: str = 'close', n: int = 14):
        """
        Get the minimum and maximum values for a given number of periods (n)
        """
        close = self.ohlcv_df[col_name]
        roll_low = close.rolling(n).min()
        roll_high = close.rolling(n).max()
        return roll_low, roll_high
    
    def moving_average(self, col_name: str = 'close', n: int = 7):
        """Calculate the simple moving average of a specified column in the dataframe

        Args:
            col_name (str, optional): column name. Defaults to 'close'.
            n (int, optional): MA trailing number. Defaults to 7.

        Returns:
            pd.Series: a series of moving average values
        """
        if isinstance(n, int):
            return self.ohlcv_df[col_name].rolling(n).mean()
        elif isinstance(n, list):
            ma_res_list = [self.ohlcv_df[col_name].rolling(i).mean() for i in n]
            ma_res_df = pd.concat(ma_res_list, axis=0)
            ma_res_df.columns = [f'ma_{i}' for i in n]
            return ma_res_df
    
    def moving_average_deviation(self, col_name: str = 'close', n: int = 7):
        """Calculate the moving average deviation. 
        Using the given column name, the function will calculate the MA using the values from that column.
        Then it will calculate the MA deviation using the formula: MA deviation = original value - MA value

        Args:
            col_name (str, optional): column name. Defaults to 'close'.
            n (int, optional): MA trailing number. Defaults to 7.

        Returns:
            pd.Series: a series of moving average deviation
        """
        if isinstance(n, int):
            ma_series = self.moving_average(col_name, n)
            return (self.ohlcv_df[col_name].iloc[n:] - ma_series) / ma_series
        elif isinstance(n, list):
            # * assert all elements are integers
            assert all(isinstance(elem, int) for elem in n), 'All elements in the input list must be integers.'

            ma_dev_df = pd.DataFrame()
            for i in n:
                ma_series = self.moving_average(col_name, i)
                ma_dev_df[f'ma_{i}_pct_deviation'] = (self.ohlcv_df[col_name].iloc[i:] - ma_series) / ma_series
            return ma_dev_df

    def rolling_sd(self, col_name: str = 'close', n: int = 14):
        if isinstance(n, int):
            sd_series = self.ohlcv_df[col_name].rolling(n).std()
        return sd_series
        pass

    def RSI(self, n: int = 14):
        """calculate the relative strength index (RSI) from a given rolling period

        Args:
            n (int, optional): number of rolling period. Defaults to 14.

        Returns:
            pd.Series: a time-series of RSI
        """
        delta = self.ohlcv_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=n).mean()
        avg_loss = loss.rolling(window=n).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)

    def stochasticRSI(self, n: int = 14, d: int = 3, concat_result: bool = True):
        """calculate stochastic RSI

        Args:
            n (int, optional): number of rolling period. Defaults to 14.
            d (int, optional): number of RSI rolling period to be calculated using the stochastic oscillator formula. Defaults to 3.

        Returns:
            pd.Series: a pandas series of stochastic RSI
        """
        rsi = self.RSI(n)
        stoch_rsi_k = (rsi - rsi.rolling(n).min()) / \
            (rsi.rolling(n).max() - rsi.rolling(n).min())
        stoch_rsi_d = stoch_rsi_k.rolling(d).mean()
        if concat_result:
            res = pd.concat([stoch_rsi_k, stoch_rsi_d], axis=1)
            res.columns = ['stoch_rsi_k', 'stoch_rsi_d']
            return res
        else:
            return stoch_rsi_k, stoch_rsi_d

    def MACD(self, n_long: int = 26, n_short: int = 12, concat_result: bool = False):
        """calculate MACD

        Args:
            n_long (int, optional): a number of long period. Defaults to 26.
            n_short (int, optional): a number of short period. Defaults to 12.

        Returns:
            pd.Series: a pandas series of MACD
        """
        assert n_long > n_short, "Number of long period should be greater than number of short period."
        ema_long = self.ohlcv_df['close'].ewm(
            span=n_long, min_periods=n_long).mean()
        ema_short = self.ohlcv_df['close'].ewm(
            span=n_short, min_periods=n_short).mean()
        macd = (ema_short - ema_long).fillna(0)
        signal = macd.ewm(span=9, min_periods=9).mean().fillna(0)
        if concat_result:
            res = pd.concat([macd, signal], axis=1)
            res.columns = ['macd', 'macd_signal']
            return res
        else:
            return macd, signal

    def bollinger_bands(self, n: int = 20, k: float = 2.0, concat_result: bool = False):
        """calculate the Bollinger Bands

        Args:
            n (int, optional): number of rolling period. Defaults to 20.
            k (float, optional): a standard deviation multiplier to create upper and lower bands. Defaults to 2.0.

        Returns:
            pd.Series: a pandas series of Bollinger Band value
        """
        rolling_mean = self.ohlcv_df['close'].rolling(window=n).mean()
        rolling_std = self.ohlcv_df['close'].rolling(window=n).std()
        upper_band = rolling_mean + (k * rolling_std)
        lower_band = rolling_mean - (k * rolling_std)
        if concat_result:
            res = pd.concat([upper_band, lower_band], axis=1)
            res.columns = ['upper_bollinger_band', 'lower_bollinger_band']
            return res
        else:
            return upper_band, lower_band

    def volume_change_pct(self, n: int = 10):
        """calculate the volume change percentage of a series by dividing the current volume with the average volume of latest n periods. this indicator can signify a spike in current volume compared to previous ones

        Args:
            n (int, optional): number of rolling period. Defaults to 10.

        Returns:
            pd.Series: a series of volume change percentage
        """
        volume = self.ohlcv_df[['volume']]
        volume['average_previous_volume'] = volume.rolling(
            n).mean().shift(1).fillna(0)
        pct_change = (volume['volume'] - volume['average_previous_volume']
                      ) / volume['average_previous_volume']

        # TODO: if pct change is inf, use max value found within n period
        pct_change = pct_change.apply(
            lambda x: float('nan') if x == float('inf') else x)
        pct_change = pct_change.fillna(pct_change.rolling(n).max()).fillna(0)
        return pct_change

    def overnight_return(self):
        """calculate overnight return which is the return from current day's open price compared to last day's close price

        Returns:
            pd.Series: a series of overnight return
        """
        prev_close = self.ohlcv_df['close'].shift(1)
        overnight_return = (self.ohlcv_df['open'] - prev_close) / prev_close
        return overnight_return

    def candlestick_volume_ratio(self, mode):
        """calculate candlestick volume ratio which is the candlestick length (high - low or open - close, depend on choosing) divided by the respective volume. if such ratio significantly changes from the previous day (we may also need to consider the absolute volume), some trend reversion may occur

        Args:
            mode (str): mode of candlestick length. if set to whisker, candlestick length is high - low. if set to body, candlestick length is open - close.

        Returns:
            pd.Series: a series of candlestick volume ratio
        """
        if mode == 'whisker':
            i, j = 'high', 'low'
        elif mode == 'body':
            i, j = 'open', 'close'
        ratio = self.ohlcv_df.apply(
            lambda x: 0 if x['volume'] == 0 else abs(
                x[i] - x[j]) / x['volume'], axis=1)
        return ratio

    def bollinger_ratio(self, n: int = 20, k: float = 2.0):
        """calculate the Bollinger ratio which follows this equation:
            bollinger_ratio = (close price - bollinger's lower band) / (bollinger's upper band - bollinger's lower band)


        Args:
            n (int, optional): number of rolling period. Defaults to 20.
            k (float, optional): a standard deviation multiplier to create upper and lower bands. Defaults to 2.0.

        Returns:
            pd.Series: a pandas series of Bollinger ratio value
        """
        upper_band, lower_band = self.bollinger_bands(n=n, k=k)
        gap = self.ohlcv_df['close'] - lower_band
        width = upper_band - lower_band
        ratio = gap / width
        return ratio

    def AROON(self, n: int = 25):
        """calculate AROON indicator

        Args:
            n (int, optional): number of rolling period. Defaults to 25.

        Returns:
            pd.Series: a series of AROON values
        """
        high = self.ohlcv_df['high'].rolling(
            n +
            1).apply(
            np.argmax,
            raw=True).fillna(n)
        low = self.ohlcv_df['low'].rolling(
            n +
            1).apply(
            np.argmin,
            raw=True).fillna(n)
        aroon_up = ((n - high) / n) * 100
        aroon_down = ((n - low) / n) * 100
        return aroon_up, aroon_down

    def stochastic_oscillator(self, n: int = 14, d: int = 3):
        """calculate stochastic oscillator value

        Args:
            n (int, optional): number of rolling period. Defaults to 14.
            d (int, optional): number of rolling period for stochastic process. Defaults to 3.

        Returns:
            pd.Series: a series of stochastic oscillator
        """
        close = self.ohlcv_df['close']
        roll_low, roll_high = self._get_min_max(n)

        # TODO: calculate %K
        k_percent = 100 * ((close - roll_low) / (roll_high - roll_low))

        # TODO: calculate %D
        d_percent = k_percent.rolling(window=d).mean()

        return k_percent, d_percent


class IndicatorExecutor():
    def __init__(self) -> None:
        pass

    def combine_indicators(self, indicator_dict: dict, ticker_name: str = None):
        res_list = []
        for k, v in indicator_dict.items():
            if isinstance(v, pd.Series):
                v = v.to_frame()
                v.columns = [k]
            elif not isinstance(v, pd.DataFrame):
                print(type(v))
                raise TypeError('only supports pandas Series and DataFrame')
            res_list.append(v)

        res_df = pd.concat(res_list, axis=1)

        # * insert ticker name
        if ticker_name:
            res_df.insert(0, 'ticker', ticker_name)
        return res_df

    def execute_object(self, obj, function_args_dict, ticker_name: str = None, concat_result: bool = True, verbose: bool = False):
        all_result = {}
        for function_name, kw_arguments in function_args_dict.items():
            if hasattr(obj, function_name) and callable(getattr(obj, function_name)):
                function_to_call = getattr(obj, function_name)
                result = function_to_call(**kw_arguments)
                all_result[function_name] = result
                if verbose:
                    print(f"Function '{function_name}' executed with {kw_arguments}")
            else:
                if verbose:
                    print(f"Function '{function_name}' does not exist or is not callable.")

        if concat_result:
            return self.combine_indicators(all_result, ticker_name)
        else:
            return all_result

    def generate_indicator_grid(self, data, indicator_params, ticker_list: list = None, ticker_col_name: str = 'ticker'):
        if not ticker_list:
            ticker_list = data['ticker'].unique()

        indicator_table_list = [
            self.execute_object(
                TechnicalIndicators(data[data[ticker_col_name] == ticker]), indicator_params, ticker, True, False
            )
            for ticker in ticker_list
        ]

        indicator_df = pd.concat(indicator_table_list, axis=0)
        return indicator_df