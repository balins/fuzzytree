"""
This module gathers fuzzy tree-based methods. Currently, only single-output
decision tree classification is supported.
"""

from abc import ABCMeta, abstractmethod
from sys import getrecursionlimit

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, _check_sample_weight

from . import _criterion
from ._fuzzy_tree import FuzzyTreeBuilder, FuzzyTree
from ._splitter import Splitter

__all__ = ["FuzzyDecisionTreeClassifier"]

# =============================================================================
# Types and constants
# =============================================================================

CRITERIA_CLF = {"gini": _criterion.gini_index,
                "entropy": _criterion.entropy_decrease,
                "misclassification": _criterion.misclassification_decrease}


# =============================================================================
# Base decision tree
# =============================================================================

class BaseFuzzyDecisionTree(BaseEstimator, metaclass=ABCMeta):
    """Base class for fuzzy decision trees.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self, *,
                 fuzziness,
                 criterion,
                 max_depth,
                 min_membership_split,
                 min_membership_leaf,
                 min_impurity_decrease):
        self.fuzziness = fuzziness
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_membership_split = min_membership_split
        self.min_membership_leaf = min_membership_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def get_depth(self):
        """Return the depth of the decision tree.
        The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.
        """
        check_is_fitted(self)
        return self.tree_.max_depth

    def get_n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            The number of leaves.
        """
        check_is_fitted(self)
        return self.tree_.n_leaves

    def _get_sample_weight(self, X, sample_weight=None):
        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        indices = np.flatnonzero(sample_weight)
        sample_weight = sample_weight[indices] / sample_weight[indices].mean()

        return indices, sample_weight

    def fit(self, X, y, sample_weight=None, check_input=True):
        if check_input:
            X, y = check_X_y(X, y, estimator=self)

        indices, sample_weight = self._get_sample_weight(X, sample_weight)
        X, y = X[indices], y[indices]
        self.X_, self.y_ = X, y
        n_samples, self.n_features_ = X.shape
        self.n_features_in_ = self.n_features_

        is_classification = is_classifier(self)

        if y.ndim != 1:
            raise ValueError("Multi-output problems are not currently supported (got y.ndim=%s)." % y.ndim)
        self.n_outputs_ = 1

        if is_classification:
            check_classification_targets(y)
            self.classes_, self.y_ = np.unique(y, return_inverse=True)
            self.n_classes_ = self.classes_.shape[0]
        else:
            raise NotImplementedError("Regression trees are not currently supported.")

        max_depth = (getrecursionlimit() if self.max_depth is None
                     else self.max_depth)

        if not self.min_membership_leaf > 0.:
            raise ValueError("min_membership_leaf must be greater than 0.0, got %s" % self.min_membership_leaf)

        if not self.min_membership_split > 0.:
            raise ValueError("min_membership_split must be greater than 0.0, got %s" % self.min_membership_split)

        min_membership_split = max(self.min_membership_split, 2 * self.min_membership_leaf)

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero")

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")

        if is_classification:
            splitter = Splitter(CRITERIA_CLF[self.criterion], self.fuzziness)
        else:
            raise NotImplementedError("Regression trees are not currently supported.")

        if is_classifier(self):
            self.tree_ = FuzzyTree(self.y_, sample_weight, self.n_classes_)
        else:
            raise NotImplementedError("Regression trees are not currently supported.")

        builder = FuzzyTreeBuilder(splitter,
                                   min_membership_split,
                                   self.min_membership_leaf,
                                   max_depth,
                                   self.min_impurity_decrease)

        builder.build(self.tree_, self.X_, self.y_, sample_weight)

        return self

    def _validate_X_predict(self, X, check_input):
        """Validate the training data on predict (probabilities)."""
        if check_input:
            X = self._validate_data(X, dtype=np.float64, reset=False)
        else:
            # The number of features is checked regardless of `check_input`
            self._check_n_features(X, reset=False)
        return X

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)

        _, sample_weight = self._get_sample_weight(X)

        proba = self.tree_.predict(X, sample_weight)

        if is_classifier(self):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)
            else:
                raise ValueError("Multi-output problems are not currently supported.")
        else:
            raise NotImplementedError("Regression trees are not currently supported.")

    def _get_tags(self):
        return {
            **super()._get_tags(),
            "requires_positive_X": False,
            "requires_y": True,
        }


# =============================================================================
# Public estimators
# =============================================================================

class FuzzyDecisionTreeClassifier(ClassifierMixin, BaseFuzzyDecisionTree):
    """A fuzzy decision tree classifier.
    Read more in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    fuzziness : float, default=1.0
        The fuzziness parameter that controls softness of the tree between 0.
        (hard) and 1. (soft).
    criterion : {"gini", "entropy", "misclassification"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity, "entropy" for the information gain and
        "misclassification" for the misclassification ratio.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_membership_split sum of membership or recursion limit is met.
    min_membership_split : float, default=2.0
        The minimum sum of membership required to split an internal node.
    min_membership_leaf : float, default=1.0
        The minimum sum of membership required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_membership_leaf`` sum of membership in each of the left and
        right branches. This may have the effect of smoothing the model,
        especially in regression.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            impurity - M_t_R / M_t * right_impurity
                     - M_t_L / M_t * left_impurity

        where ``M_t_L`` is the sum of membership in the left child and ``M_t_R``
        is the sum of membership in the right child.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed. Currently, always equals 1.
    tree_ : FuzzyTree
        The underlying Tree object. Please refer to
        ``help(fuzzytree._fuzzy_tree.FuzzyTree)`` for attributes of Tree object and
        for basic usage of these attributes.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_membership_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.
    The :meth:`predict` method operates using the :func:`numpy.argmax`
    function on the outputs of :meth:`predict_proba`. This means that in
    case the highest predicted probabilities are tied, the classifier will
    predict the tied class with the lowest index in :term:`classes_`.
    """

    def __init__(self,
                 fuzziness=1.,
                 criterion='gini',
                 max_depth=None,
                 min_membership_split=2.,
                 min_membership_leaf=1.,
                 min_impurity_decrease=0.):
        super().__init__(
            fuzziness=fuzziness,
            criterion=criterion,
            max_depth=max_depth,
            min_membership_split=min_membership_split,
            min_membership_leaf=min_membership_leaf,
            min_impurity_decrease=min_impurity_decrease
        )

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a fuzzy decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings. Internally,
            it will be converted to an array-like of integers.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights used as initial membership function for samples.
            If None, then samples are equally weighted. Splits
            that would create child nodes with net zero membership are
            ignored while searching for a split in each node.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : FuzzyDecisionTreeClassifier
            Fitted estimator.
        """
        return super().fit(X, y, sample_weight, check_input)

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.
        The predicted class probability is the fraction of membership of samples
        of the same class in a leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)

        _, sample_weight = self._get_sample_weight(X)

        proba = self.tree_.predict(X, sample_weight)

        if is_classifier(self):
            if self.n_outputs_ == 1:
                return proba
            else:
                raise ValueError("Multi-output problems are not currently supported.")
        else:
            raise NotImplementedError("Regression trees are not currently supported.")

    def predict_log_proba(self, X, check_input=True):
        """Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        proba = self.predict_proba(X, check_input)

        if is_classifier(self):
            if self.n_outputs_ == 1:
                return np.log(proba)
            else:
                raise ValueError("Multi-output problems are not currently supported.")
        else:
            raise NotImplementedError("Regression trees are not currently supported.")

    def _get_tags(self):
        return {
            **super()._get_tags(),
            "binary_only": False
        }
