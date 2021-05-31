import numpy as np

from ._utils import split_by_membership


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

    def build(self, X, y, n_outputs):
        self.n_outputs = n_outputs
        membership = np.ones(X.shape[0])
        return self._build(X, y, membership, 0)

    def _build(self, X, y, membership, depth):
        if depth == self.max_depth or X.shape[0] < self.min_samples_split:
            return FuzzyTree(y, membership, self.n_outputs)

        rule, gain = self.splitter.node_split(X, y, membership)
        if not rule or gain < self.min_impurity_decrease:
            return FuzzyTree(y, membership, self.n_outputs)

        new_membership = rule.evaluate(X)
        membership_true, indices_true = split_by_membership(membership, new_membership)
        membership_false, indices_false = split_by_membership(membership, 1. - new_membership)

        if not abs(membership.sum() - (membership_true.sum() + membership_false.sum())) < 5:
            print(membership_true.sum() + membership_false.sum())
            print(membership)
            print(membership_false)
            print(membership_true)
            raise ValueError(membership.sum())

        if min(membership_false.sum(), membership_true.sum()) < self.min_samples_leaf:
            return FuzzyTree(y, membership, self.n_outputs)

        true_branch = self._build(X[indices_true], y[indices_true], membership_true, depth + 1)
        false_branch = self._build(X[indices_false], y[indices_false], membership_false, depth + 1)

        return FuzzyTree(y, membership, self.n_outputs, rule, true_branch, false_branch)


class FuzzyTree:
    def __init__(self, y, membership, n_outputs, rule=None, true_branch=None, false_branch=None):
        self.membership = np.array([np.sum(membership[y == cls]) for cls in range(n_outputs)])
        self.membership /= self.membership.sum()
        self.n_outputs = n_outputs
        self.rule = rule
        self.true_branch = true_branch
        self.false_branch = false_branch

    def predict(self, X, membership):
        if self.is_leaf:
            return self._predict_leaf(membership)
        else:
            return self._predict_internal(X, membership)

    def _predict_internal(self, X, membership):
        new_membership = self.rule.evaluate(X)
        membership_true, indices_true = split_by_membership(membership, new_membership)
        membership_false, indices_false = split_by_membership(membership, 1. - new_membership)

        prediction = np.zeros((X.shape[0], self.n_outputs))

        if indices_true.size > 0:
            prediction[indices_true] = self.true_branch.predict(X[indices_true], membership_true)
        if indices_false.size > 0:
            prediction[indices_false] += self.false_branch.predict(X[indices_false], membership_false)

        return prediction

    def _predict_leaf(self, membership):
        return membership[:, None] * self.membership

    @property
    def is_leaf(self):
        return self.rule is None
