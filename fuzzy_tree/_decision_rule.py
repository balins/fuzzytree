import numpy as np


class DecisionRule:
    def __init__(self, feature, boundary, is_categorical):
        self.feature = feature
        self.boundary = boundary
        self.is_categorical = is_categorical

        if self.is_categorical:
            rule = self.boundary.__contains__
        else:
            rule = self.boundary.__lt__

        self.evaluate_ = np.vectorize(rule)

    def evaluate(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if n_features < self.feature - 1:
            raise ValueError(f"Input of this decision rule must have at least {self.feature + 1} features."
                             f"Got {n_features}")
        if n_samples == 0:
            return np.array([], dtype=bool)

        return self.evaluate_(X[:, self.feature])

    def __str__(self):
        return "x[%s] %s %s?" % (self.feature, 'in' if self.is_categorical else '>=', self.boundary)

    def __repr__(self):
        return str(self)
