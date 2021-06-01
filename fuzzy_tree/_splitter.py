import numpy as np

from ._decision_rule import DecisionRule


class Splitter:
    def __init__(self, gain_function):
        self.gain_function_ = gain_function

    def node_split(self, X, y, membership, fuzziness):
        best_gain = 0.
        best_rule = None

        if np.unique(y).shape[0] <= 1:
            return best_rule, best_gain

        for feature_idx in range(X.shape[1]):
            sorted_ = np.sort(np.unique(X[:, feature_idx]))
            midpoint_splits = ((sorted_ + np.roll(sorted_, -1)) / 2)[:-1]

            for split_val in midpoint_splits:
                rule = DecisionRule(sorted_, split_val, feature_idx, fuzziness)
                new_membership = rule.evaluate(X)
                gain = self.gain_function_(y, membership, new_membership)
                if gain > best_gain:
                    best_rule, best_gain = rule, gain

        return best_rule, best_gain
