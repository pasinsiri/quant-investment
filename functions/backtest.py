import pandas as pd 
import numpy as np
import patsy
import tqdm
from statsmodels.formula.api import ols

class BacktestPreparator():
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
    
    def wins(x, lower:float, upper:float):
        return np.where(x <= lower, lower, np.where(x >= upper, upper, x))
    
    def get_formula(factors, Y):
        L = ["0"]
        L.extend(factors)
        return f'{Y} ~ {" + ".join(L)}'
    
    def factors_from_names(n):
        return list(filter(lambda x: "USFASTD_" in x, n))

    def estimate_factor_returns(self, df, return_col:str = 'DlyReturn'):     
        # * winsorize returns for fitting 
        df[return_col] = self.wins(df[return_col], -0.25, 0.25)
    
        all_factors = self.factors_from_names(list(df))
        form = self.get_formula(all_factors, return_col)
        model = ols(form, data=df)
        results = model.fit()
        return results
    
    def setdiff(base:list, exclude:list):
        exs = set(exclude)
        res_list = [x for x in base if x not in exs]
        return res_list
    
    def model_matrix(formula, data): 
        _, predictors = patsy.dmatrices(formula, data)
        return predictors
    
    def colnames(B):
        if type(B) == patsy.design_info.DesignMatrix:
            return B.design_info.column_names
        elif type(B) == pd.core.frame.DataFrame:
            return B.columns.tolist()
        return None