import pandas as pd 
import numpy as np 

class TechnicalIndicators():
    def __init__(self) -> None:
        pass
    
    # * moving correlation
    def corr_over_time(self, s1:pd.Series, s2:pd.Series, start_n:int = 10, window_mode:str = 'rolling'):
        """calculate correlations over time. note that both series should have datetime as their indices, and it should be the same for each row

        Args:
            s1 (pd.Series): the first series
            s2 (pd.Series): the other one
            start_n (int, optional): number of backward rows used to calculate correlation. Defaults to 10.
        """

        if not s1.index.equals(s2.index):
            raise ValueError('indices of s1 and s2 do not match each other')

        out = pd.Series(index=s1.index.tolist()[start_n:])
        if window_mode == 'expanding':
            for n in range(start_n, s1.shape[0]):
                tmp_s1 = s1.iloc[:n]
                tmp_s2 = s2.iloc[:n]
                out.iloc[n - start_n] = tmp_s1.corr(tmp_s2)
        elif window_mode == 'rolling':
            for n in range(start_n, s1.shape[0]):
                tmp_s1 = s1.iloc[n - start_n:n]
                tmp_s2 = s2.iloc[n - start_n:n]
                out.iloc[n - start_n] = tmp_s1.corr(tmp_s2)
                pass 
        else:
            raise ValueError('window_mode can be either rolling or expanding')
        
        return out

    # * relative strength indicator (RSI)
    def rsi(self, prices:pd.Series, n_period:int = 50):
        """calculate RSI for the given prices' series

        Args:
            prices (pd.Series): a time-seires of prices
            n_period (int, optional): number of intervals. Defaults to 50.
        """

        # check if n_period <= series' length - 1 (exclude the first row since we need to find price differences before calculating RSI)
        if n_period > prices.shape[0] - 1:
            raise ValueError('n_period is greather than the series length')

        res = []
        indices = []
        delta = prices.diff().iloc[1:]

        # use moving window in this case, note that we can use expand window
        for i in range(delta.shape[0] - n_period):
            tmp = delta.iloc[i: i + n_period]
            current_index = tmp.index[-1]
            pos_mean = np.mean([x for x in tmp.tolist() if x > 0])
            neg_mean = -1 * np.mean([x for x in tmp.tolist() if x < 0]) # because the change itself is negative
            rsi = (pos_mean / neg_mean) / (1 + (pos_mean / neg_mean))
            res.append(rsi)
            indices.append(current_index)

        rsi_series = pd.Series(res, index = indices)
        return rsi_series

    # * moving average convergence divergence (MACD)
    def macd(self, prices:pd.Series, n_short:int = 12, n_long:int = 26):
        prices = prices.to_frame()
        prices['ma_short'] = prices['Close'].ewm(span = n_short, adjust = False, min_periods = n_short).mean()
        prices['ma_long'] = prices['Close'].ewm(span = n_long, adjust = False, min_periods = n_long).mean()
        prices['macd'] = prices.apply(lambda x: x['ma_short'] - x['ma_long'], axis = 1)
        return prices[['macd']]