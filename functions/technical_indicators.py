import pandas as pd 
import numpy as np 

class TechnicalIndicators():
    def __init__(self) -> None:
        pass 
    
    def corr_over_time(s1, s2, start_n = 10):
        """calculate correlations over time. note that both series should have datetime as their indices, and it should be the same for each row

        Args:
            s1 (pd.Series): the first series
            s2 (pd.Series): the other one
            start_n (int, optional): number of backward rows used to calculate correlation. Defaults to 10.
        """

        if s1.index != s2.index:
            raise ValueError('indices of s1 and s2 do not match each other')

        # Create output series
        out = pd.Series(index=s1.index.tolist()[start_n:])

        # We use expand window in this case, note that we can also use moving window
        for n in range(start_n, s1.shape[0]):
            tmp_s1 = s1.iloc[:n]
            tmp_s2 = s2.iloc[:n]

            out.iloc[n - start_n] = tmp_s1.corr(tmp_s2)
        del tmp_s1, tmp_s2
        return out