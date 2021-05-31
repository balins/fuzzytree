import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._criterion import gini_criterion, entropy_decrease, misclassification_decrease
from ._splitter import Splitter
from ._tree import FuzzyDecisionTreeBuilder

CRITERION_CLF = {
    'gini': gini_criterion,
    'entropy': entropy_decrease,
    'misclassification': misclassification_decrease
}


class FuzzyDecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """A fuzzy decision tree classifier
    .
    Read more in the :ref:`User Guide <fuzzy_tree>`.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=None,
                 min_impurity_decrease=0.):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, y, check_input=True):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : object
            Returns self.
            :param check_input:
        """
        if check_input:
            X, y = check_X_y(X, y)

        self.X_ = X
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.max_features_ = X.shape[1]
        self.n_features_ = X.shape[1]

        self.builder_ = FuzzyDecisionTreeBuilder(splitter=Splitter(CRITERION_CLF[self.criterion]),
                                                 max_depth=self.max_depth,
                                                 min_samples_split=self.min_samples_split,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 min_impurity_decrease=self.min_impurity_decrease,
                                                 )

        self.tree_ = self.builder_.build(self.X_, self.y_, self.n_classes_)

        return self

    def _predict(self, X, check_input=True):
        """Predict log probabilities for each class each samples in X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array-like of (n_samples, n_outputs)
            The predicted log probabilities of classes.
        """
        check_is_fitted(self)
        if check_input:
            X = check_array(X)

        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('Number of input features is different from what was seen '
                             f'in `fit` (was {self.X_.shape[1]}, got {X.shape[1]})')

        prediction = self.tree_.predict(X, np.ones(X.shape[0]))

        if np.any(prediction.sum(axis=1) > 1.01) or np.any(prediction.sum(axis=1) < 0.97):
            raise ValueError(prediction.sum(axis=1))

        return prediction

    def predict(self, X):
        membership = self._predict(X)
        return self.classes_[np.argmax(membership, axis=1)]

    def predict_proba(self, X):
        return self._predict(X)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def _get_tags(self):
        return {
            'multilabel': True,
            'requires_y': True
        }
