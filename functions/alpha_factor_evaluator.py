import pandas as pd 
import numpy as np
import alphalens as al

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