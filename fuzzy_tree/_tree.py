import numpy as np

from ._utils import split_by_membership, membership_ratio


class FuzzyTreeBuilder:
    def __init__(self,
                 splitter,
                 fuzziness,
                 min_membership_split,
                 min_membership_leaf,
                 max_depth,
                 min_impurity_decrease):
        self.splitter = splitter
        self.fuzziness = fuzziness
        self.min_membership_split = min_membership_split
        self.min_membership_leaf = min_membership_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    def build(self, tree, X, y, membership):
        self._build(tree, X, y, membership, 0)

    def _build(self, tree, X, y, membership, depth):
        if depth >= self.max_depth or membership.sum() < self.min_membership_split:
            return

        rule, gain = self.splitter.node_split(X, y, membership, self.fuzziness)
        if not rule or gain < self.min_impurity_decrease:
            return

        new_membership = rule.evaluate(X)
        membership_true, indices_true = split_by_membership(membership, new_membership)
        membership_false, indices_false = split_by_membership(membership, 1. - new_membership)

        if min(membership_false.sum(), membership_true.sum()) < self.min_membership_leaf:
            return

        tree.rule = rule
        tree.true_branch = FuzzyTree(y[indices_true], membership_true, tree.n_classes, depth + 1)
        tree.false_branch = FuzzyTree(y[indices_false], membership_false, tree.n_classes, depth + 1)

        self._build(tree.true_branch, X[indices_true], y[indices_true], membership_true, depth + 1)
        self._build(tree.false_branch, X[indices_false], y[indices_false], membership_false, depth + 1)


class FuzzyTree:
    def __init__(self, y, membership, n_classes, level=0, rule=None, true_branch=None, false_branch=None):
        self.class_weights = membership_ratio(y, membership, np.arange(n_classes))
        self.n_classes = n_classes
        self.level = level
        self.rule = rule
        self.true_branch = true_branch
        self.false_branch = false_branch

    def predict(self, X, membership):
        return (self._predict_leaf(membership) if self.is_leaf
                else self._predict_internal(X, membership))

    def _predict_internal(self, X, membership):
        new_membership = self.rule.evaluate(X)
        membership_true, indices_true = split_by_membership(membership, new_membership)
        membership_false, indices_false = split_by_membership(membership, 1. - new_membership)

        prediction = np.zeros((X.shape[0], self.n_classes))

        if indices_true.size > 0:
            prediction[indices_true] = self.true_branch.predict(X[indices_true], membership_true)
        if indices_false.size > 0:
            prediction[indices_false] += self.false_branch.predict(X[indices_false], membership_false)

        return prediction

    def _predict_leaf(self, membership):
        return membership[:, None] * self.class_weights

    @property
    def max_depth(self):
        return (self.level if self.is_leaf
                else max(self.true_branch.max_depth, self.false_branch.max_depth))

    @property
    def n_leaves(self):
        return (1 if self.is_leaf
                else self.true_branch.n_leaves + self.false_branch.n_leaves)

    @property
    def is_leaf(self):
        return self.rule is None
