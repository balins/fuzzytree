import numpy as np
from more_itertools import powerset

from ._decision_rule import DecisionRule


def split_node(X, y, gain_function):
    best_gain = 0.
    best_rule = None
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if np.unique(y).shape[0] <= 1:
        return best_rule, best_gain

    for feature in range(n_features):
        categorical = not np.issubdtype(X[:, feature].dtype, np.number)
        if categorical:
            unique = X[:, feature].unique()
            splits = map(np.array, filter(lambda subset: 0 < len(subset) < len(unique), powerset(unique)))
        else:
            sorted_ = np.sort(X[:, feature])
            splits = map(lambda i: (sorted_[i] + sorted_[i + 1]) / 2, range(n_samples - 1))

        for split in splits:
            rule = DecisionRule(feature, split, categorical)
            mask = rule.evaluate(X)
            gain = gain_function(X, y, mask)
            if gain > best_gain:
                best_rule, best_gain = rule, gain

    return best_rule, best_gain
