import numpy as np
from ._utils import s_shaped_membership, sigmoid_membership


class DecisionRule:
    def __init__(self, sorted_feature, split_val, feature_idx):
        self.split_val = split_val
        self.feature_idx = feature_idx

        # todo: choose membership -> s-shaped (default), sigmoid
        # todo: choose crispness (b/c attributes in memb. functions)

        min_, max_ = sorted_feature[0], sorted_feature[-1]
        a = split_val - sorted_feature.std()
        b = split_val + sorted_feature.std()

        self.universe = np.linspace(min_, max_, 100)
        self.membership = s_shaped_membership(self.universe, a, b)
        # self.membership = sigmoid_membership(self.universe, split_val, sorted_feature.std())

        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots(figsize=(8, 9))
        #
        # ax.plot(self.universe, 1 - self.membership, 'b', linewidth=1.5, label='Low')
        # ax.plot(self.universe, self.membership, 'r', linewidth=1.5, label='High')
        # ax.set_title(f'feature {feature_idx}, split {split_val}')
        # ax.legend()
        #
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.get_xaxis().tick_bottom()
        # ax.get_yaxis().tick_left()
        #
        # plt.show()

    def evaluate(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if n_features < self.feature_idx - 1:
            raise ValueError(f"Input of this decision rule must have at least {self.feature_idx + 1} features."
                             f"Got {n_features}")
        if n_samples == 0:
            return np.array([])

        return self.evaluate_(X[:, self.feature_idx])

    def evaluate_(self, x):
        return np.interp(x, self.universe, self.membership)

    def __str__(self):
        return "x[%s] >= %s?" % (self.feature_idx, self.split_val)

    def __repr__(self):
        return str(self)
