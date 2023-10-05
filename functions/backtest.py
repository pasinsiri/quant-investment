"""
A python module for continuous backtesting, with its own optimization function and its respective gradient

Author: pasinsiri
Initialized Date: 2023-07-18
"""

import pandas as pd
import numpy as np
import patsy
import scipy
from statsmodels.formula.api import ols
from tqdm import tqdm

class Backtest():
    """perform backtesting on a given time-series data of asset price
    """
    def __init__(self, factor_df, covariance, return_df, alpha_factors:list, risk_factors:list, n_forward_return:int, risk_aversion_coefficient:float, date_index:int = 1) -> None:
        self.factor_df = factor_df
        self.covariance = covariance
        self.return_df = return_df
        self.alpha_factors = alpha_factors
        self.risk_factors = risk_factors
        self.risk_aversion_coefficient = risk_aversion_coefficient
        self.date_index = date_index

        self.join_df = self._map_forward_return(n_forward_return=n_forward_return)
        self.df_dict_by_date = self._split_to_date(self.join_df, date_index=self.date_index)
        self.factor_return_df = {date: self.estimate_factor_returns(self.df_dict_by_date[date]).params for date in self.df_dict_by_date}

    def _split_to_date(self, df, date_index:int):
        dates = df.index.levels[date_index].tolist()
        frames = dict()
        for d in dates:
            date_df = df[df.index.get_level_values(date_index) == d]
            frames[d] = date_df
        return frames

    def _map_forward_return(self, n_forward_return:int):
        """map a return data to the factor data in a forward manner

        Args:
            n_forward_return (int): number of periods in advanced the return will be used

        Returns:
            pd.DataFrame: a mapped dataframe
        """
        data = self.factor_df.copy()
        self.return_df = self.return_df.rename(columns={'date': 'return_date'})
        dates_df = self.return_df[['return_date']].sort_values(by='return_date').drop_duplicates().reset_index(drop=True)
        dates_df['date'] = dates_df['return_date'].shift(n_forward_return)
        self.return_df = self.return_df.merge(dates_df, on='return_date', how='left')
        self.return_df = self.return_df.set_index(['date', 'Barrid'])

        data.set_index([data.index, 'Barrid'], inplace=True)
        join_df = data.merge(self.return_df, left_index=True, right_index=True, how='left')
        return join_df
    
    def _winsorize(self, x, lower:float, upper:float):
        """winsorize the array

        Args:
            x (np.array): a numpy array
            lower (float): a lower bound
            upper (float): an upper bound

        Returns:
            np.array: an array of winsorized values
        """
        return np.where(x <= lower, lower, np.where(x >= upper, upper, x))
    
    def _clean_nas(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for numeric_column in numeric_columns:
            df[numeric_column] = np.nan_to_num(df[numeric_column])
        
        return df
    
    def get_formula(self, factors:list, Y:str):
        """generate a string of formula

        Args:
            factors (list): a list of factors
            Y (str): a dependent variable

        Returns:
            str: a string of formula
        """
        L = ["0"]
        L.extend(factors)
        return f'{Y} ~ {" + ".join(L)}'
    
    # def factors_from_names(self, n, factor_keyword:str = 'USFASTD_'):
    #     return list(filter(lambda x: factor_keyword in x, n))


    def estimate_factor_returns(self, df, return_col:str = 'DlyReturn', lower_bound:float = -0.25, upper_bound:float = 0.25, factor_keyword:str = 'USFASTD_'):
        """estimate the factor return values

        Args:
            df (pd.DataFrame): a combined pandas dataframe of factor exposures and return
            return_col (str): return's column name
            lower_bound (float, optional): a lower bound of which the return will be winsorized. Defaults to -0.25.
            upper_bound (float, optional): an upper bound of which the return will be winsorized. Defaults to 0.25.
            factor_keyword: a keyword use to identify factor columns in the dataframe. Defualts to USFASTD_

        Returns:
            pd.DataFrame: a pandas dataframe of OLS-fitted values
        """
        # * winsorize returns for fitting 
        df[return_col] = self._winsorize(df[return_col], lower_bound, upper_bound)
    
        # all_factors = self.factors_from_names(list(df))
        all_factors = list(filter(lambda x: factor_keyword in x, list(df)))
        form = self.get_formula(all_factors, return_col)
        model = ols(form, data=df)
        results = model.fit()
        return results
    
    # def setdiff(self, base:list, exclude:list):
    #     exs = set(exclude)
    #     res_list = [x for x in base if x not in exs]
    #     return res_list
    
    def model_matrix(self, formula, data):
        """generate patsy dmatrix from a given data and formula

        Args:
            formula (str): a formula string
            data (pd.DataFrame): a dataframe

        Returns:
            patsy.dmatrices: a patsy dmatrices
        """
        _, predictors = patsy.dmatrices(formula, data)
        return predictors
    
    def colnames(self, B) -> list:
        """get column names from a given dataFrame

        Args:
            B (pandas dataframe or patsy dmatrices): a given data

        Returns:
            list: a list of column names
        """
        if isinstance(B, patsy.design_info.DesignMatrix):
            return B.design_info.column_names
        if isinstance(B, pd.core.frame.DataFrame):
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
    
    def get_lambda(self, universe, composite_volume_column:str = 'ADTCA_30'):
        """calculate lambda (the transaction cost)

        Args:
            universe (pd.DataFrame): a time-series dataframe which contains the composite volume column
            composite_volume_column (str, optional): a composite volumn column. Defaults to 'ADTCA_30'.

        Returns:
            pd.Series: a time-series of estimated transaction costs
        """
        # TODO: lambda is transaction cost
        adv = universe[[composite_volume_column]]
        adv.loc[np.isnan(adv[composite_volume_column]), composite_volume_column] = 1.0e4
        adv.loc[adv[composite_volume_column] == 0, composite_volume_column] = 1.0e4 
        return 0.1 / adv

    def get_B_alpha(self, universe):
        # TODO: Implement
        return self.model_matrix(self.get_formula(self.alpha_factors, 'SpecRisk'), data = universe)
    
    # def get_alpha_vec(self, B_alpha):
    #     """
    #     Create an alpha vecrtor

    #     Parameters
    #     ----------        
    #     B_alpha : patsy.design_info.DesignMatrix 
    #         Matrix of Alpha Factors
            
    #     Returns
    #     -------
    #     alpha_vec : patsy.design_info.DesignMatrix 
    #         alpha vector
    #     """
        
    #     # TODO: Implement
    #     return 1e-4 * np.sum(B_alpha, axis = 1)
        
    def get_obj_func(self, h0, Q, specVar, alpha_vec, Lambda): 
        def obj_func(h):
            # TODO: Implement
            Qh = Q @ h
            factor_risk = 0.5 * self.risk_aversion_coefficient * np.sum(Qh ** 2)
            idiosyncratic_risk = 0.5 * self.risk_aversion_coefficient * ((h ** 2) @ specVar)
            expected_return = alpha_vec.T @ h
            transaction_costs = ((h - h0) ** 2) @ Lambda
            
            return - expected_return + factor_risk + idiosyncratic_risk + transaction_costs
        
        return obj_func
    
    def get_grad_func(self, h0, Q, specVar, alpha_vec, Lambda):
        def grad_func(h):
            # TODO: Implement
            gradient = (self.risk_aversion_coefficient * (Q.transpose() @ (Q @ h))) + \
                        (self.risk_aversion_coefficient * specVar * h) - alpha_vec + \
                        (2 * (h - h0) * Lambda)
            
            return np.asarray(gradient)
        
        return grad_func
    
    def get_h_star(self, Q, specVar, alpha_vec, h0, Lambda):
        """
        Optimize the objective function

        Parameters
        ----------        
        risk_aversion : int or float 
            Trader's risk aversion
            
        Q : patsy.design_info.DesignMatrix 
            Q Matrix
            
        specVar: Pandas Series 
            Specific Variance
            
        alpha_vec: patsy.design_info.DesignMatrix 
            alpha vector
            
        h0 : Pandas Series  
            initial holdings
            
        Lambda : Pandas Series  
            Lambda
            
        Returns
        -------
        optimizer_result[0]: Numpy ndarray 
            optimized holdings
        """
        obj_func = self.get_obj_func(h0, Q, specVar, alpha_vec, Lambda)
        grad_func = self.get_grad_func(h0, Q, specVar, alpha_vec, Lambda)
        
        # TODO: Implement 
        optimizer_result = scipy.optimize.fmin_l_bfgs_b(obj_func, h0, fprime = grad_func)
        return optimizer_result[0]
    
    def get_risk_exposures(self, B, h_star):
        """
        Calculate portfolio's Risk Exposure

        Parameters
        ----------
        B : patsy.design_info.DesignMatrix 
            Matrix of Risk Factors
            
        h_star: Numpy ndarray 
            optimized holdings
            
        Returns
        -------
        risk_exposures : Pandas Series
            Risk Exposures
        """
        
        # TODO: Implement
        return pd.Series(B.transpose() @ h_star, index = B.design_info.column_names)
    
    def form_optimal_portfolio(self, df, previous):
        df = df.reset_index(level=0).merge(previous, how = 'left', on = 'Barrid')
        df = self._clean_nas(df)
        df.loc[df['SpecRisk'] == 0]['SpecRisk'] = np.median(df['SpecRisk'])
    
        # universe = self.get_universe(df).reset_index()
        universe = df.reset_index()
        date = universe['date'][1]    
        h0 = universe['h.opt.previous']
    
        B = self.model_matrix(self.get_formula(self.risk_factors, "SpecRisk"), universe)
        BT = B.transpose()
    
        specVar = (0.01 * universe['SpecRisk']) ** 2
        Fvar = self.diagonal_factor_cov(date, B)
        
        Lambda = self.get_lambda(universe).values.reshape(-1)
        B_alpha = self.get_B_alpha(universe)
        alpha_vec = 1e-4 * np.sum(B_alpha, axis = 1)
    
        Q = np.matmul(scipy.linalg.sqrtm(Fvar), BT)
        
        h_star = self.get_h_star(Q, specVar, alpha_vec, h0, Lambda)
        opt_portfolio = pd.DataFrame(data = {"Barrid" : universe['Barrid'], "h.opt" : h_star})
        
        risk_exposures = self.get_risk_exposures(B, h_star)
        portfolio_alpha_exposure = self.get_portfolio_alpha_exposure(B_alpha, h_star)
        total_transaction_costs = self.get_total_transaction_costs(h0, h_star, Lambda)
    
        return {
            "opt.portfolio" : opt_portfolio, 
            "risk.exposures" : risk_exposures, 
            "alpha.exposures" : portfolio_alpha_exposure,
            "total.cost" : total_transaction_costs}
    
    def get_portfolio_alpha_exposure(self, B_alpha, h_star):
        """
        Calculate portfolio's Alpha Exposure

        Parameters
        ----------
        B_alpha : patsy.design_info.DesignMatrix 
            Matrix of Alpha Factors
            
        h_star: Numpy ndarray 
            optimized holdings
            
        Returns
        -------
        alpha_exposures : Pandas Series
            Alpha Exposures
        """
        
        # TODO: Implement
        
        return pd.Series(np.matmul(B_alpha.transpose(), h_star), index = self.colnames(B_alpha))
        
    def get_total_transaction_costs(self, h0, h_star, Lambda):
        """
        Calculate Total Transaction Costs

        Parameters
        ----------
        h0 : Pandas Series
            initial holdings (before optimization)
            
        h_star: Numpy ndarray 
            optimized holdings
            
        Lambda : Pandas Series  
            Lambda
            
        Returns
        -------
        total_transaction_costs : float
            Total Transaction Costs
        """
        
        # TODO: Implement
        
        return np.sum(Lambda * ((h_star - h0) ** 2))
    
    def build_tradelist(self, prev_holdings, opt_result):
        tmp = prev_holdings.merge(opt_result['opt.portfolio'], how='outer', on = 'Barrid')
        tmp['h.opt.previous'] = np.nan_to_num(tmp['h.opt.previous'])
        tmp['h.opt'] = np.nan_to_num(tmp['h.opt'])
        return tmp
    
    def convert_to_previous(self, result):
        prev = result['opt.portfolio']
        prev = prev.rename(index=str, columns={"h.opt": "h.opt.previous"}, copy=True, inplace=False)
        return prev
    
    def run_backtest(self, frames:dict, previous_holdings:pd.DataFrame):
        trades = {}
        port = {}

        for date in tqdm(frames.keys(), desc='Optimizing Portfolio', unit='day'):
            frame_df = frames[date]
            result = self.form_optimal_portfolio(frame_df, previous_holdings)
            trades[date] = self.build_tradelist(previous_holdings, result)
            port[date] = result
            previous_holdings = self.convert_to_previous(result)

        return {
            'trades': trades,
            'port': port
        }

class SimpleBacktest():
    def __init__(self, factor_df, price_df, n_offset: int = 1, allow_overlap: bool = False, trading_cost_in: float = 0.0, trading_cost_out: float = 0.0) -> None:
        self.factor_df = factor_df
        self.price_df = price_df
        self.n_offset = n_offset
        self.allow_overlap = allow_overlap
        self.trading_cost_in = trading_cost_in
        self.trading_cost_out = trading_cost_out
        self.signal_df = None

    def generate_signal(self, signal_function: callable, func_arg: dict, **kwargs):
        if not callable(signal_function):
            raise TypeError('signal_function must be callable.')
        
        self.signal_df = signal_function(self.factor_df)