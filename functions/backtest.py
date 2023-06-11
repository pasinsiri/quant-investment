import pandas as pd 
import numpy as np
import patsy
import tqdm

class BackTestPrep():
    def __init__(self, factor_df, covariance, return_df, alpha_factors:list, risk_factors:list, **kwargs) -> None:
        self.factor_df = factor_df 
        self.covariance = covariance 
        self.return_df = return_df

    def map_forward_return(self, n_forward_return:int):
        data = self.factor_df.copy()
        self.return_df = self.return_df.rename(columns={'date': 'return_date'})
        dates_df = self.return_df[['return_date']].sort_values(by='return_date').drop_duplicates().reset_index(drop=True)
        dates_df['date'] = dates_df['return_date'].shift(n_forward_return)
        self.return_df = self.return_df.merge(dates_df, on='return_date', how='left')
        self.return_df = self.return_df.set_index(['date', 'Barrid'])

        data.set_index([data.index, 'Barrid'], inplace=True)
        join_df = data.merge(self.return_df, left_index=True, right_index=True, how='left')
        return join_df