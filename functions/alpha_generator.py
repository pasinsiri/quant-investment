import abc
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.base import clone, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
from typing import Union, Optional, List, Literal

class NoOverlapClassifierAbstract(VotingClassifier):
    @abc.abstractmethod
    def _calculate_oob_score(self, classifiers: BaseEstimator):
        raise NotImplementedError

    @abc.abstractmethod
    def _non_overlapping_estimators(
        self, 
        x: pd.DataFrame, 
        y: pd.Series, 
        classifiers: List[BaseEstimator], 
        n_skip_samples: int
    ):
        raise NotImplementedError

    def __init__(
        self,
        estimator,
        voting: Literal['hard', 'soft'] = 'soft',
        n_skip_samples: int = 4
    ):
        # List of estimators for all the subsets of data
        estimators = [('clf' + str(i), estimator)
                      for i in range(n_skip_samples + 1)]

        self.n_skip_samples = n_skip_samples
        super().__init__(estimators=estimators, voting=voting)

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        weights: Optional[List[Union[int, float]]] = None
    ):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(
            X, y, clone_clfs, self.n_skip_samples)
        self.named_estimators_ = Bunch(
            **dict(zip(estimator_names, self.estimators_)))
        self.oob_score_ = self._calculate_oob_score(self.estimators_)

        return self


class NoOverlapClassifier(NoOverlapClassifierAbstract):
    def __init__(
            self,
            estimator,
            voting: str = 'soft',
            n_skip_samples: int = 4):
        self.estimator = estimator
        super().__init__(
            estimator=self.estimator,
            voting=voting,
            n_skip_samples=n_skip_samples)

    def _non_overlapping_samples(self, x, y, n_skip_samples, start_i=0):
        """
        Get the non overlapping samples.

        Parameters
        ----------
        x : DataFrame
            The input samples
        y : Pandas Series
            The target values
        n_skip_samples : int
            The number of samples to skip
        start_i : int
            The starting index to use for the data

        Returns
        -------
        non_overlapping_x : 2 dimensional Ndarray
            The non overlapping input samples
        non_overlapping_y : 1 dimensional Ndarray
            The non overlapping target values
        """
        assert len(x.shape) == 2
        assert len(y.shape) == 1

        x_dates = x.index.levels[0][start_i::n_skip_samples + 1]
        y_dates = y.index.levels[0][start_i::n_skip_samples + 1]
        non_overlapping_x = x[x.index.get_level_values(0).isin(x_dates)]
        non_overlapping_y = y[y.index.get_level_values(0).isin(y_dates)]

        return non_overlapping_x, non_overlapping_y

    def _calculate_oob_score(self, classifiers):
        """
        Calculate the mean out-of-bag score from the classifiers.

        Parameters
        ----------
        classifiers : list of Scikit-Learn Classifiers
            The classifiers used to calculate the mean out-of-bag score

        Returns
        -------
        oob_score : float
            The mean out-of-bag score
        """
        oob_score = np.mean([clf.oob_score_ for clf in classifiers])
        return oob_score

    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        """
        Fit the classifiers to non overlapping data.

        Parameters
        ----------
        x : DataFrame
            The input samples
        y : Pandas Series
            The target values
        classifiers : list of Scikit-Learn Classifiers
            The classifiers used to fit on the non overlapping data
        n_skip_samples : int
            The number of samples to skip

        Returns
        -------
        fit_classifiers : list of Scikit-Learn Classifiers
            The classifiers fit to the the non overlapping data
        """

        n_classifiers = len(classifiers)
        samples = [
            self._non_overlapping_samples(
                x,
                y,
                n_skip_samples,
                start_i) for start_i in range(n_classifiers)]
        fit_classifiers = [
            classifiers[i].fit(
                sample[0],
                sample[1]) for i,
            sample in enumerate(samples)]

        return fit_classifiers
