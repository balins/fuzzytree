import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._criterion import gain_ratio, gini_criterion, misclassification_ratio
from ._splitter import split_node
from ._tree import DecisionNode, Leaf

CRITERION_CLF = {
    'gini': gini_criterion,
    'entropy': gain_ratio,
    'misclassification': misclassification_ratio
}


class FuzzyDecisionTreeBuilder:
    def __new__(cls, clf, X, y, depth=0):
        params = clf.get_params()

        if depth == params['max_depth'] or X.shape[0] < params['min_samples_split']:
            return Leaf(y, clf.classes_)

        if params['criterion'] in CRITERION_CLF:
            gain_function = CRITERION_CLF[params['criterion']]
        else:
            gain_function = params['criterion']

        rule, gain = split_node(X, y, gain_function)

        if not rule or gain < params['min_impurity_decrease']:
            return Leaf(y, clf.classes_)

        mask = rule.evaluate(X)
        n_true = np.count_nonzero(mask)
        n_false = mask.shape[0] - n_true

        if min(n_true, n_false) < params['min_samples_leaf']:
            return Leaf(y, clf.classes_)

        true_branch = FuzzyDecisionTreeBuilder(clf, X[mask], y[mask], depth + 1)
        false_branch = FuzzyDecisionTreeBuilder(clf, X[~mask], y[~mask], depth + 1)

        return DecisionNode(rule, true_branch, false_branch, clf.classes_)


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

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)

        self.classes_, y = np.unique(y, return_inverse=True)

        self.X_ = X
        self.y_ = y

        self.decision_node_ = FuzzyDecisionTreeBuilder(self, self.X_, self.y_)

        return self

    def predict_(self, X, log_proba):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        check_is_fitted(self, 'decision_node_')

        X = check_array(X)
        X_T = np.transpose(X)

        if X_T.shape[0] != self.X_.T.shape[0]:
            raise ValueError('Number of input features is different from what was seen'
                             f'in `fit` (was {self.X_.T.shape[0]}, got {X_T.shape[0]})')

        y = self.decision_node_.predict_log_proba(X)

        if log_proba:
            return y
        else:
            return self.classes_[np.argmax(y, axis=1)]

    def predict(self, X):
        return self.predict_(X, log_proba=False)

    def predict_proba(self, X):
        return np.exp(self.predict_(X, log_proba=True))

    def predict_log_proba(self, X):
        return self.predict_(X, log_proba=True)

    def _get_tags(self):
        return {
            'multilabel': True,
            'pairwise': True,
            'requires_y': True
        }
