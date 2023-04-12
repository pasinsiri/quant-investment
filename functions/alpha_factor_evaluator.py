import pandas as pd 
import numpy as np
import alphalens as al
import matplotlib.pyplot as plt 

class AlphaFactorEvaluator():
    def __init__(self, factor_return, price) -> None:
        self.factor_return = factor_return
        self.factor_names = factor_return.columns
        self.price = price

    def combine_factor_forward_returns(self, period:int, max_loss:float, verbose:bool = False):
        factor_data_dict = dict()
        for factor in self.factor_names:
            if verbose: print(f'Formatting factor data for {factor}')
            factor_data_dict[factor] = al.utils.get_clean_factor_and_forward_returns(
                factor = self.factor_return[factor],
                prices = self.price,
                periods = [period],
                max_loss = max_loss
            )
        return factor_data_dict
    
    def get_factor_returns(self, factor_data_dict):
        factor_return_list = []
        for factor in self.factor_names:
            factor_return = al.performance.factor_returns(factor_data_dict[factor])
            factor_return.columns = [factor]
            factor_return_list.append(factor_return)
        return pd.concat(factor_return_list, axis = 1)
    
    # TODO: factor evaluation
    # * Sharpe ratio
    def _sharpe_ratio(df, frequency:str):
        if frequency == "daily":
            annualization_factor = np.sqrt(252)
        elif frequency == "monthly":
            annualization_factor = np.sqrt(12)
        else:
            annualization_factor = 1
            
        sharpe_ratio = annualization_factor * (df.mean() / df.std())
        
        return sharpe_ratio
    
    def get_sharpe_ratio(factor_return_df, frequency:str = 'daily'):
        return factor_return_df.apply(factor_return_df, frequency = frequency, axis = 0)

    # * information coefficient
    def get_information_coefficient(self, factor_data_dict)
        rank_ic_list = []

        for factor in self.factor_names:
            rank_ic = al.performance.factor_information_coefficient(factor_data_dict[factor])
            rank_ic.columns = [factor]
            rank_ic_list.append(rank_ic)

        return pd.concat(rank_ic_list, axis = 0)
