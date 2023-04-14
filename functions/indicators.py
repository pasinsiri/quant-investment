import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self, ohlcv_df):
        self.ohlcv_df = ohlcv_df
    
    def rsi(self, n:int = 14):
        delta = self.ohlcv_df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=n).mean()
        avg_loss = loss.rolling(window=n).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def macd(self, n_long:int = 26, n_short:int = 12):
        assert n_long > n_short, "Number of long period should be greater than number of short period."
        ema_long = self.ohlcv_df['Close'].ewm(span=n_long, min_periods=n_long).mean()
        ema_short = self.ohlcv_df['Close'].ewm(span=n_short, min_periods=n_short).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=9, min_periods=9).mean()
        return macd, signal
    
    def bollinger_bands(self, n:int = 20, k = 2):
        rolling_mean = self.ohlcv_df['Close'].rolling(window=n).mean()
        rolling_std = self.ohlcv_df['Close'].rolling(window=n).std()
        upper_band = rolling_mean + (k * rolling_std)
        lower_band = rolling_mean - (k * rolling_std)
        return upper_band, lower_band
    
    def volume_change_pct(self):
        volume = self.ohlcv_df['Volume']
        pct_change = volume.pct_change()
        return pct_change
    
    def overnight_return(self):
        prev_close = self.ohlcv_df['Close'].shift(1)
        overnight_return = (self.ohlcv_df['Open'] - prev_close) / prev_close
        return overnight_return
    
    def candlestick_volume_ratio(self):
        candlestick_range = self.ohlcv_df['High'] - self.ohlcv_df['Low']
        ratio = candlestick_range / self.ohlcv_df['Volume']
        return ratio
    
    def bollinger_ratio(self, n:int = 20, k:int = 2):
        upper_band, lower_band = self.bollinger_bands(n = n, k = k)
        gap = self.ohlcv_df['Close'] - lower_band
        width = upper_band - lower_band
        ratio = gap / width
        return ratio
