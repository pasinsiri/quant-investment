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
        self.alpha_factors = alpha_factors
        self.risk_factors = risk_factors

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
    
    def get_formula(self, factors, Y):
        L = ["0"]
        L.extend(factors)
        return f'{Y} ~ {" + ".join(L)}'
    
    def factors_from_names(self, n):
        return list(filter(lambda x: "USFASTD_" in x, n))

    def estimate_factor_returns(self, df, return_col:str = 'DlyReturn'):     
        # * winsorize returns for fitting 
        df[return_col] = self.wins(df[return_col], -0.25, 0.25)
    
        all_factors = self.factors_from_names(list(df))
        form = self.get_formula(all_factors, return_col)
        model = ols(form, data=df)
        results = model.fit()
        return results
    
    def setdiff(self, base:list, exclude:list):
        exs = set(exclude)
        res_list = [x for x in base if x not in exs]
        return res_list
    
    def model_matrix(self, formula, data): 
        _, predictors = patsy.dmatrices(formula, data)
        return predictors
    
    def colnames(B):
        if type(B) == patsy.design_info.DesignMatrix:
            return B.design_info.column_names
        elif type(B) == pd.core.frame.DataFrame:
            return B.columns.tolist()
        return None

    def diagonal_factor_cov(self, date, B):
        """
        Create the factor covariance matrix

        Parameters
        ----------
        date : string
            date. For example 20040102
            
        B : patsy.design_info.DesignMatrix OR pandas.core.frame.DataFrame
            Matrix of Risk Factors
            
        Returns
        -------
        Fm : Numpy ndarray
            factor covariance matrix    
        """
        
        # TODO: Implement
        cov = self.covariance[self.covariance.index == date]
        k = np.shape(B)[1] # number of risk factors
        factor_names = self.colnames(B)
        factor_cov = np.zeros([k, k])
        for i in range(k):
            cov_value = cov[(cov['Factor1'] == factor_names[i]) & (cov['Factor2'] == factor_names[i])]['VarCovar']
            factor_cov[i, i] = 1e-4 * cov_value # convert to decimal since the raw data comes in the pct squared format
        return factor_cov
    
    def get_lambda(self, composite_volume_column:str = 'ADTCA_30'):
        # TODO: lambda is transaction cost
        adv = self.factor_df[composite_volume_column]
        adv.loc[np.isnan(adv[composite_volume_column]), composite_volume_column] = 1.0e4
        adv.loc[adv[composite_volume_column] == 0, composite_volume_column] = 1.0e4 
        return 0.1 / adv

    def get_B_alpha(self):
        # TODO: Implement
        return self.model_matrix(self.get_formula(self.alpha_factors, 'SpecRisk'), data = self.factor_df)
    
    def get_alpha_vec(B_alpha):
        """
        Create an alpha vecrtor

        Parameters
        ----------        
        B_alpha : patsy.design_info.DesignMatrix 
            Matrix of Alpha Factors
            
        Returns
        -------
        alpha_vec : patsy.design_info.DesignMatrix 
            alpha vector
        """
        
        # TODO: Implement
        return 1e-4 * np.sum(B_alpha, axis = 1)
