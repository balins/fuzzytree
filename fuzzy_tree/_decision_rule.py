import numpy as np

import skfuzzy as fuzz


class DecisionRule:
    def __init__(self, sorted_feature, split_val, feature_idx):
        self.split_val = split_val
        self.feature_idx = feature_idx

        min_ = sorted_feature[0]
        max_ = sorted_feature[-1]
        margin = min(split_val - min_, max_ - split_val) / 2
        self.universe = sorted_feature

        self.membership = 1. - fuzz.trapmf(self.universe, [min_, min_, split_val - margin, split_val + margin])

        if not 0.48 < fuzz.interp_membership(self.universe, self.membership, split_val, zero_outside_x=False) < 0.52:
            raise ValueError(fuzz.interp_membership(self.universe, self.membership, split_val, zero_outside_x=False))

        # print("split_val")
        # print(split_val)
        # print("universe")
        # print(self.universe)
        # print("membership")
        # print(self.membership)
        # print("linspace")
        # print(np.linspace(min_, max_, 8))
        # print("memb")
        # print(fuzz.interp_membership(self.universe, self.membership, np.linspace(min_, max_, 8), zero_outside_x=False))
        # print('------!!!-------')

    def evaluate(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if n_features < self.feature_idx - 1:
            raise ValueError(f"Input of this decision rule must have at least {self.feature_idx + 1} features."
                             f"Got {n_features}")
        if n_samples == 0:
            return np.array([[], [], []], dtype=np.float32)

        return self.evaluate_(X[:, self.feature_idx])

    def evaluate_(self, x):
        return fuzz.interp_membership(self.universe, self.membership, x, zero_outside_x=False)

    def __str__(self):
        return "x[%s] >= %s?" % (self.feature_idx, self.split_val)

    def __repr__(self):
        return str(self)
