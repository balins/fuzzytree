import numpy as np

from ._decision_rule import DecisionRule
from ._utils import powerset, midpoints


class Splitter:
    def __init__(self, gain_function):
        self.gain_function_ = gain_function

    def node_split(self, X, y):
        best_gain = 0.
        best_rule = None
        n_features = X.shape[1]

        if np.unique(y).shape[0] <= 1:
            return best_rule, best_gain

        for ith_feature in range(n_features):
            feature = X[:, ith_feature]
            categorical = not np.issubdtype(feature.dtype, np.number)
            if categorical:
                splits = powerset(feature)
            else:
                splits = midpoints(feature)

            for split in splits:
                rule = DecisionRule(ith_feature, split, categorical)
                mask = rule.evaluate(X)
                gain = self.gain_function_(X, y, mask)
                if gain > best_gain:
                    best_rule, best_gain = rule, gain

        return best_rule, best_gain
