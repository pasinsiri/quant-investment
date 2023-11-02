import numpy as np
import pandas as pd
import cvxpy as cvx
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod

# TODO: Risk Model with PCA


class RiskModelPCA():
    def __init__(
            self,
            returns,
            annualization_factor,
            num_factor_exposures,
            svd_solver: str = 'full'):
        self.returns = returns
        self.ticker_list = returns.columns
        self.return_dates = returns.index
        self.pca_factor_names = [
            f'PC_{i}' for i in range(num_factor_exposures)]
        self.annualization_factor = annualization_factor
        self.pca = PCA(
            n_components=num_factor_exposures,
            svd_solver=svd_solver).fit(returns)

        self.factor_betas_ = self.get_factor_beta()
        self.factor_returns_ = self.get_factor_return()
        self.factor_cov_matrix_ = self.get_factor_cov_matrix()
        self.idiosyncratic_var_matrix_ = self.get_idiosyncratic_var_matrix()

    def get_factor_beta(self):
        return pd.DataFrame(
            self.pca.components_.T,
            index=self.ticker_list,
            columns=self.pca_factor_names)

    def get_factor_return(self):
        return pd.DataFrame(
            self.pca.transform(
                self.returns),
            index=self.return_dates,
            columns=self.pca_factor_names)

    def get_idiosyncratic_var_matrix(self):
        common_returns_ = pd.DataFrame(
            np.dot(
                self.factor_returns_,
                self.factor_betas_.T),
            index=self.return_dates,
            columns=self.ticker_list)
        residuals_ = (self.returns - common_returns_)
        return pd.DataFrame(
            np.diag(
                np.var(residuals_)) *
            self.annualization_factor,
            index=self.ticker_list,
            columns=self.ticker_list)

    def get_factor_cov_matrix(self):
        return np.diag(
            self.factor_returns_.var(
                axis=0,
                ddof=1) *
            self.annualization_factor)


# TODO: Portfolio Holding Optimizer
# * abstract object
class AbstractOptimalHoldings(ABC):
    @abstractmethod
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """

        raise NotImplementedError()
        return obj

    @abstractmethod
    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """

        raise NotImplementedError()

    def _get_risk(
            self,
            weights,
            factor_betas,
            alpha_vector_index,
            factor_cov_matrix,
            idiosyncratic_var_vector):
        f = factor_betas.loc[alpha_vector_index].values.T * weights
        X = factor_cov_matrix
        S = np.diag(
            idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())

        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)

    def find(
            self,
            alpha_vector,
            factor_betas,
            factor_cov_matrix,
            idiosyncratic_var_vector):
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(
            weights,
            factor_betas,
            alpha_vector.index,
            factor_cov_matrix,
            idiosyncratic_var_vector)

        obj = self._get_obj(weights, alpha_vector)
        constraints = self._get_constraints(
            weights, factor_betas.loc[alpha_vector.index].values, risk)

        prob = cvx.Problem(obj, constraints)
        prob.solve(max_iters=500)

        optimal_weights = np.asarray(weights.value).flatten()

        return pd.DataFrame(data=optimal_weights, index=alpha_vector.index)

# * normal / regularization optimizer


class OptimalHoldings(AbstractOptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert len(alpha_vector.columns) == 1, 'alpha_vector should be vector'
        assert self.lambda_reg >= 0, 'Lambda value cannot be negative'
        if self.lambda_reg == 0.0:
            obj = cvx.Minimize(-1 * alpha_vector.values.flatten() * weights)
        else:
            normal_term = -1 * alpha_vector.values.T * weights
            regularized_term = cvx.norm(weights, 2) * self.lambda_reg
            obj = cvx.Minimize(normal_term + regularized_term)

        return obj

    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """
        assert (len(factor_betas.shape) == 2)
        constraints = [
            risk <= self.risk_cap ** 2,
            factor_betas.T * weights <= self.factor_max,
            factor_betas.T * weights >= self.factor_min,
            sum(cvx.abs(weights)) <= 1.0,
            sum(weights) == 0.0,
            weights <= self.weights_max,
            weights >= self.weights_min
        ]

        return constraints

    def __init__(
            self,
            lambda_reg: float = 0.0,
            risk_cap: float = 0.05,
            factor_max: float = 10.0,
            factor_min: float = -10.0,
            weights_max: float = 0.55,
            weights_min: float = -0.55):
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min
        self.lambda_reg = lambda_reg

# * holding-stricted optimizer (minimized volatility compared to a specific list of weights)


class OptimalHoldingsStrictFactor(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert (len(alpha_vector.columns) == 1)

        # TODO: Implement function
        weights_star = (alpha_vector.values.flatten(
        ) - np.mean(alpha_vector.values.flatten())) / np.sum(np.abs(alpha_vector.values.flatten()))
        obj = cvx.Minimize(cvx.norm(weights - weights_star, 2))

        return obj
