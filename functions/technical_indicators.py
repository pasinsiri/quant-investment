import pandas as pd 
import numpy as np 

class TechnicalIndicators():
    def __init__(self) -> None:
        pass 
    
    # * moving correlation
    def corr_over_time(self, s1:pd.Series, s2:pd.Series, n:int = 10, window_mode:str = 'rolling'):
        """calculate correlations over time. note that both series should have datetime as their indices, and it should be the same for each row

        Args:
            s1 (pd.Series): the first series
            s2 (pd.Series): the other one
            n (int, optional): number of backward rows used to calculate correlation. Defaults to 10.
        """

        # ! edit later
        # if s1.index != s2.index:
        #     raise ValueError('indices of s1 and s2 do not match each other')

        out = pd.Series(index=s1.index.tolist()[n:])
        if window_mode == 'expanding':
            for n in range(n, s1.shape[0]):
                tmp_s1 = s1.iloc[:n]
                tmp_s2 = s2.iloc[:n]
                out.iloc[n - n] = tmp_s1.corr(tmp_s2)
        elif window_mode == 'rolling':
            for n in range(n, s1.shape[0]):
                tmp_s1 = s1.iloc[n - n:n]
                tmp_s2 = s2.iloc[n - n:n]
                out.iloc[n - n] = tmp_s1.corr(tmp_s2)
                pass 
        else:
            raise ValueError('window_mode can be either rolling or expanding')
        
        return out

    # * relative strength indicator (RSI)
    def rsi(prices:pd.Series, n_period:int = 50):
        """calculate RSI for the given prices' series

        Args:
            prices (pd.Series): a time-seires of prices
            n_period (int, optional): number of intervals. Defaults to 50.
        """

        # check if n_period <= series' length - 1 (exclude the first row since we need to find price differences before calculating RSI)
        if n_period > prices.shape[0] - 1:
            raise ValueError('n_period is greather than the series length')

        rsi_series = pd.Series()
        delta = prices.diff().iloc[1:]

        # use moving window in this case, note that we can use expand window
        for i in range(n_period, delta.shape[0]):
            tmp = delta.iloc[i: i + n_period]
            current_index = tmp.index[-1]
            pos_mean = np.mean([x for x in tmp.tolist() if x > 0])
            neg_mean = -1 * np.mean([x for x in tmp.tolist() if x < 0]) # because the change itself is negative
            rsi = (pos_mean / neg_mean) / (1 + (pos_mean / neg_mean))

            rsi_series = rsi_series.set_value(current_index, rsi)
        return rsi_series
