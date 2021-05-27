import warnings

import numpy as np


class FuzzyDecisionTreeBuilder:
    def __init__(self,
                 splitter,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 min_impurity_decrease):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def build(self, X, y, all_classes):
        self.all_classes_ = all_classes
        return self._build(X, y, 0)

    def _build(self, X, y, depth):
        if depth == self.max_depth or X.shape[0] < self.min_samples_split:
            return FuzzyTree(y, self.all_classes_)

        rule, gain = self.splitter.node_split(X, y)

        if not rule or gain < self.min_impurity_decrease:
            return FuzzyTree(y, self.all_classes_)

        mask = rule.evaluate(X)
        n_true = np.count_nonzero(mask)
        n_false = mask.shape[0] - n_true

        if min(n_true, n_false) < self.min_samples_leaf:
            return FuzzyTree(y, self.all_classes_)

        true_branch = self._build(X[mask], y[mask], depth + 1)
        false_branch = self._build(X[~mask], y[~mask], depth + 1)

        return FuzzyTree(y, self.all_classes_, rule, true_branch, false_branch)


class FuzzyTree:
    def __init__(self, classes, all_classes, rule=None, true_branch=None, false_branch=None):
        self.all_classes = all_classes
        self.counts_ = np.bincount(classes, minlength=self.all_classes.shape[0])
        self.rule = rule
        self.true_branch = true_branch
        self.false_branch = false_branch

    def predict(self, X):
        if self.is_leaf:
            return self._predict_leaf(X)
        else:
            return self._predict_internal(X)

    def _predict_internal(self, X):
        mask = self.rule.evaluate(X)
        prediction = np.zeros((mask.shape[0], self.all_classes.shape[0]))

        n_true = np.count_nonzero(mask)
        n_false = X.shape[0] - n_true

        if n_true > 0:
            prediction[mask] = self.true_branch.predict(X[mask])
        if n_false > 0:
            prediction[~mask] = self.false_branch.predict(X[~mask])

        return prediction

    def _predict_leaf(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classes_log_proba = np.log(self.counts_) - np.log(np.sum(self.counts_))

        prediction = np.array([classes_log_proba] * X.shape[0])

        return prediction

    @property
    def is_leaf(self):
        return self.rule is None
