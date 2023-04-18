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
        """utilize the alphalens library to combine a dataframe of factor return for each price and each date to a dataframe of stock price.

        Args:
            period (int): a forward period of which the return will be calculated
            max_loss (float): max loss to be parsed to alphalens
            verbose (bool, optional): if set to True, steps will be printed. Defaults to False.

        Returns:
            dict: a dictionary of which keys represent factor name and values represent factor and forward returns 
        """
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
        """utilize the alphalens library to calculate factor returns from factor values

        Args:
            factor_data_dict (dict): the result from the combine_factor_forward_returns function

        Returns:
            pd.DataFrame: a dataframe of which rows represent the dates and columns represent the factors
        """
        factor_return_list = []
        for factor in self.factor_names:
            factor_return = al.performance.factor_returns(factor_data_dict[factor])
            factor_return.columns = [factor]
            factor_return_list.append(factor_return)
        return pd.concat(factor_return_list, axis = 1)
    
    # TODO: factor evaluation
    # * Sharpe ratio
    def _sharpe_ratio(df, frequency:str):
        """calculate sharpe ratio of given factors

        Args:
            df (pd.DataFrame): a dataframe of factor values in each time period where indices represent dates and columns represent factors
            frequency (str): a frequency multiplier, can be either daily or monthly

        Returns:    
            pd.DataFrame: a dataframe consisting of sharpe ratio of each factor
        """
        if frequency == "daily":
            annualization_factor = np.sqrt(252)
        elif frequency == "monthly":
            annualization_factor = np.sqrt(12)
        else:
            annualization_factor = 1
            
        sharpe_ratio = annualization_factor * (df.mean() / df.std())
        
        return sharpe_ratio
    
    def get_sharpe_ratio(self, factor_return_df, frequency:str = 'daily'):
        """apply _sharpe_ratio function to a dataframe

        Args:
            factor_return_df (pd.DataFrame): the result from the get_factor_return function
            frequency (str, optional): a frequency multiplier, can be daily or monthly. Defaults to 'daily'.

        Returns:
            pd.DataFrame: a dataframe of sharpe ratio
        """
        return factor_return_df.apply(self._sharpe_ratio, frequency = frequency, axis = 1)

    # * information coefficient
    def get_information_coefficient(self, factor_data_dict):
        """utilize the alphalens library to calculate the rank information coefficient

        Args:
            factor_data_dict (dict): the result from the combine_factor_forward_returns function

        Returns:
            pd.DataFrame: a dataframe of which rows represent dates, columns represent factor names, and values represent rank IC of each factor in each date
        """
        rank_ic_list = []

        for factor in self.factor_names:
            rank_ic = al.performance.factor_information_coefficient(factor_data_dict[factor])
            rank_ic.columns = [factor]
            rank_ic_list.append(rank_ic)

        return pd.concat(rank_ic_list, axis = 1)
    
    # * factor rank autocorrelation (used as a proxy for portfolio turnover)
    def get_factor_rank_autocorrelation(self, factor_data_dict):
        """utilize the alphalens library to calculate the factor rank autocorrelation

        Args:
            factor_data_dict (dict): the result from the combine_factor_forward_returns function

        Returns:
            pd.DataFrame: a dataframe of which rows represent dates, columns represent factor names, and values represent rank autocorrelation of each factor in each date
        """
        rank_ac_list = []

        for factor in self.factor_names:
            rank_ac = al.performance.factor_rank_autocorrelation(factor_data_dict[factor]).to_frame()
            rank_ac.columns = [factor]
            rank_ac_list.append(rank_ac)

        return pd.concat(rank_ac_list, axis = 1)
    
    def get_mean_return_by_quantile(self, factor_data_dict):
        """utilize the alphalens library to calculate the factor's mean return by quantile

        Args:
            factor_data_dict (dict): the result from the combine_factor_forward_returns function

        Returns:
            pd.DataFrame: a dataframe of which rows represent dates, columns represent factor names, and values represent mean return by quantile of each factor in each date
        """
        qt_return_list = []

        for factor in self.factor_names:
            qt_ret, _ = al.performance.mean_return_by_quantile(factor_data_dict[factor])
            qt_ret.columns = [factor]
            qt_return_list.append(qt_ret)

        return pd.concat(qt_return_list, axis = 1)