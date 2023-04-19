import pandas as pd
import numpy as np

class TechnicalIndicators():
    def __init__(self, ohlcv_df):
        self.ohlcv_df = ohlcv_df

    def _get_min_max(self, n:int = 14):
        """
        Get the minimum and maximum values for a given number of periods (n)
        """
        min_periods = n
        close = self.ohlcv_df['close']
        roll_low = close.rolling(min_periods = min_periods, window = n).min()
        roll_high = close.rolling(min_periods = min_periods, window = n).max()
        return roll_low, roll_high
    
    def RSI(self, n:int = 14):
        delta = self.ohlcv_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=n).mean()
        avg_loss = loss.rolling(window=n).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def stochasticRSI(self, n:int = 14, k:int = 3, d:int = 3):
        rsi = self.RSI(n)
        stoch_rsi_k = (rsi - rsi.rolling(n).min()) / (rsi.rolling(n).max() - rsi.rolling(n).min())
        stoch_rsi_d = stoch_rsi_k.rolling(d).mean()
        return stoch_rsi_k, stoch_rsi_d
    
    def MACD(self, n_long:int = 26, n_short:int = 12):
        assert n_long > n_short, "Number of long period should be greater than number of short period."
        ema_long = self.ohlcv_df['close'].ewm(span=n_long, min_periods=n_long).mean()
        ema_short = self.ohlcv_df['close'].ewm(span=n_short, min_periods=n_short).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=9, min_periods=9).mean()
        return macd, signal
    
    def bollinger_bands(self, n:int = 20, k = 2):
        rolling_mean = self.ohlcv_df['close'].rolling(window=n).mean()
        rolling_std = self.ohlcv_df['close'].rolling(window=n).std()
        upper_band = rolling_mean + (k * rolling_std)
        lower_band = rolling_mean - (k * rolling_std)
        return upper_band, lower_band
    
    def volume_change_pct(self):
        volume = self.ohlcv_df['volume']
        pct_change = volume.pct_change()
        return pct_change
    
    def overnight_return(self):
        prev_close = self.ohlcv_df['close'].shift(1)
        overnight_return = (self.ohlcv_df['open'] - prev_close) / prev_close
        return overnight_return
    
    def candlestick_volume_ratio(self):
        candlestick_range = self.ohlcv_df['high'] - self.ohlcv_df['low']
        ratio = candlestick_range / self.ohlcv_df['volume']
        return ratio
    
    def bollinger_ratio(self, n:int = 20, k:int = 2):
        upper_band, lower_band = self.bollinger_bands(n = n, k = k)
        gap = self.ohlcv_df['close'] - lower_band
        width = upper_band - lower_band
        ratio = gap / width
        return ratio
    
    def AROON(self, n:int = 25):
        high = self.ohlcv_df['high'].rolling(n+1).apply(np.argmax, raw = True).fillna(n)
        low = self.ohlcv_df['low'].rolling(n+1).apply(np.argmin, raw = True).fillna(n)
        aroon_up = ((n - high) / n) * 100
        aroon_down = ((n - low) / n) * 100
        return aroon_up, aroon_down
    
    def stochastic_oscillator(self, n:int = 14, d:int = 3):
        """
        Calculate the Stochastic Oscillator indicator
        """
        close = self.ohlcv_df['close']
        roll_low, roll_high = self._get_min_max(n)
        
        # TODO: calculate %K
        k_percent = 100 * ((close - roll_low) / (roll_high - roll_low))
        
        # TODO: calculate %D
        d_percent = k_percent.rolling(window=d).mean()
        
        return k_percent, d_percent
