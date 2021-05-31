import numpy as np


class DecisionRule:
    def __init__(self, min_, split_val, max_, feature_idx):
        self.split_val = split_val
        self.feature_idx = feature_idx

        margin = min(split_val - min_, max_ - split_val) / 2
        a = split_val - margin
        b = split_val + margin
        self.universe = np.array([min_, a, split_val, b, max_])

        self.membership = np.piecewise(self.universe,
                                       [self.universe < a, self.universe >= a, self.universe >= b],
                                       [0., lambda x: (x - a) / (b - a), 1.])

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

        if not 0.49 < np.interp(split_val, self.universe, self.membership) < 0.51:
            raise ValueError(np.interp(split_val, self.universe, self.membership))

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
            return np.array([])

        return self.evaluate_(X[:, self.feature_idx])

    def evaluate_(self, x):
        return np.interp(x, self.universe, self.membership)

    def __str__(self):
        return "x[%s] >= %s?" % (self.feature_idx, self.split_val)

    def __repr__(self):
        return str(self)
