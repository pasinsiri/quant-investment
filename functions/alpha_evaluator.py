"""
Module: Alpha Factor Evaluator
Description: evaluate the alpha factor in terms of contribution to the forward return of a given asset
Author: pasinsiri
Date: 2023-07-20
"""

import pandas as pd
import numpy as np
import alphalens as al
import matplotlib.pyplot as plt

class AlphaFactorEvaluator():
    def __init__(self, factor_return, price) -> None:
        self.factor_return = factor_return
        self.factor_names = factor_return.columns
        self.price = price

    def combine_factor_forward_returns(self, periods:tuple, max_loss:float, verbose:bool = False):
        """utilize the alphalens library to combine a dataframe of factor return for each price and each date to a dataframe of stock price.

        Args:
            periods (tuple): a tuple of forward periods of which the return will be calculated. For example, if we are interested in only a single period, e.g. 1 day forward return, use (1) as input.
            max_loss (float): max loss to be parsed to alphalens
            verbose (bool, optional): if set to True, steps will be printed. Defaults to False.

        Returns:
            dict: a dictionary of which keys represent factor name and values represent factor and forward returns 
        """
        factor_data_dict = {}
        for factor in self.factor_names:
            if verbose:
                print(f'Formatting factor data for {factor}')
            factor_data_dict[factor] = al.utils.get_clean_factor_and_forward_returns(
                factor = self.factor_return[factor],
                prices = self.price,
                periods = periods,
                max_loss = max_loss
            )
        return factor_data_dict
    
    def get_factor_weights(self, factor_data_dict:dict, demeaned:bool = False, group_adjust:bool = False, equal_weight:bool = False):
        """utilize the alphalens library to compute factor-wise asset weights in a specific period. The weights is calculated by normalizing the factor values of a given factor (dividing by the absolute sum of such factor values in a given time). Please note that if all factor values in the period are positive. The weights will range from zero to one and the sum of all weights will be one. Otherwise it may not being so.

        Args:
            factor_data_dict (dict): a dictionary of a single factor dataframes with factor name as keys and factor value as values (you can obtain this dict from compute_factor_forward_returns)
            demeaned (bool, optional): should this computation happen on a long short portfolio? if True, weights are computed by demeaning factor values and dividing by the sum of their absolute value (achieving gross leverage of 1). The sum of positive weights will be the same as the negative weights (absolute value), suitable for a dollar neutral long-short portfolio. Defaults to False
            group_adjust (bool, optional): should this computation happen on a group neutral portfolio? If True, compute group neutral weights: each group will weight the same and if 'demeaned' is enabled the factor values demeaning will occur on the group level. Defaults to False
            equal_weight (bool, optional): if True the assets will be equal-weighted instead of factor-weighted. If demeaned is True then the factor universe will be split in two equal sized groups, top assets with positive weights and bottom assets with negative weights. Defaults to False

        Returns:
            pd.DataFrame: a multi-index dataframe of asset weights
        """
        factor_weight_list = []
        for factor in self.factor_names:
            factor_weight = al.performance.factor_weights(
                factor_data=factor_data_dict[factor],
                demeaned=demeaned,
                group_adjust=group_adjust,
                equal_weight=equal_weight
            ).to_frame()
            factor_weight.columns = [factor]
            factor_weight_list.append(factor_weight)

        return pd.concat(factor_weight_list, axis=1)

    def get_factor_returns(self, factor_data_dict, by_asset:bool = False, demeaned:bool = False, group_adjust:bool = False, equal_weight:bool = False):
        """utilize the alphalens library to calculate factor returns from factor values

        Args:
            factor_data_dict (dict): the result from the combine_factor_forward_returns function
            by_asset (bool, optional): if True, returns are reported separately for each esset. Defaults to False
            demeaned (bool, optional): should this computation happen on a long short portfolio? if True, weights are computed by demeaning factor values and dividing by the sum of their absolute value (achieving gross leverage of 1). The sum of positive weights will be the same as the negative weights (absolute value), suitable for a dollar neutral long-short portfolio. Defaults to False
            group_adjust (bool, optional): should this computation happen on a group neutral portfolio? If True, compute group neutral weights: each group will weight the same and if 'demeaned' is enabled the factor values demeaning will occur on the group level. Defaults to False
            equal_weight (bool, optional): if True the assets will be equal-weighted instead of factor-weighted. If demeaned is True then the factor universe will be split in two equal sized groups, top assets with positive weights and bottom assets with negative weights. Defaults to False


        Returns:
            pd.DataFrame: a dataframe of which rows represent the dates and columns represent the factors
        """
        factor_return_list = []
        for factor in self.factor_names:
            factor_return = al.performance.factor_returns(factor_data_dict[factor], demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight, by_asset=by_asset)
            factor_return.columns = [factor]
            factor_return_list.append(factor_return)
        return pd.concat(factor_return_list, axis = 1)

    # TODO: factor evaluation
    # * Sharpe ratio
    def _sharpe_ratio(self, df, frequency:str):
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
    
    # * alpha and beta of a factor
    def get_factor_alpha_beta(self, factor_data_dict:dict, demeaned:bool = False, group_adjust:bool = False, equal_weight:bool = False):
        """calculate alpha and beta values of each factor using the ordinary least square (OLS) method

        Args:
            factor_data_dict (dict): a dictionary with factor names as keys and dataframes of factor values as values (you can get this dict from combine_factor_forward_returns)
            demeaned (bool, optional): should this computation happen on a long short portfolio? if True, weights are computed by demeaning factor values and dividing by the sum of their absolute value (achieving gross leverage of 1). The sum of positive weights will be the same as the negative weights (absolute value), suitable for a dollar neutral long-short portfolio. Defaults to False
            group_adjust (bool, optional): should this computation happen on a group neutral portfolio? If True, compute group neutral weights: each group will weight the same and if 'demeaned' is enabled the factor values demeaning will occur on the group level. Defaults to False
            equal_weight (bool, optional): if True the assets will be equal-weighted instead of factor-weighted. If demeaned is True then the factor universe will be split in two equal sized groups, top assets with positive weights and bottom assets with negative weights. Defaults to False

        Returns:
            _type_: _description_
        """
        res_list = []
        for factor in self.factor_names:
            alpha_beta = al.performance.factor_alpha_beta(
                factor_data=factor_data_dict[factor],
                demeaned=demeaned,
                group_adjust=group_adjust,
                equal_weight=equal_weight
            )
            alpha_beta.columns = [f'{factor}_return_{c}' for c in alpha_beta.columns]
            res_list.append(alpha_beta)
        return pd.concat(res_list, axis=1).T
    
    def get_quantile_turnover(self, factor_data_dict:dict, n_quantile:int):
        """compute quantile turnover for each quantile in each factor

        Args:
            factor_data_dict (dict): a dictionary of factor data (you can get this dict from combine_factor_forward_returns)
            n_quantile (int): number of quantiles

        Returns:
            dict: a dictionary with factor names as keys (like factor_data_dict) and quantile turnover dataframes as values. The quantile turnover dataframe is a time-series dataframe with each column representing percentage of quantile turnover for each quantile
        """
        quantile_turnover_dict = {}
        for factor in self.factor_names:
            turnover_list = []
            for q in range(1, n_quantile):
                quantile_turnover = al.performance.quantile_turnover(factor_data_dict[factor]['factor_quantile'], quantile=q, period=1) \
                                        .to_frame()
                quantile_turnover.columns = [f'q{q}']
                turnover_list.append(quantile_turnover)
            turnover_df = pd.concat(turnover_list, axis=1) \
                            .dropna()
            quantile_turnover_dict[factor] = turnover_df
        return quantile_turnover_dict