import numpy as np

from ._fuzzy_decision_rule import FuzzyDecisionRule


class Splitter:
    """Splitter for finding the best split.

    Parameters
    ----------
    gain_function : callable
        The function to measure the quality of a split.
        Higher values indicate better split.
    fuzziness : float
        The fuzziness parameter between 0. (crisp) and 1. (fuzzy) split.
    """

    def __init__(self, gain_function, fuzziness):
        self.gain_function = gain_function
        self.fuzziness = fuzziness

    def node_split(self, X, y, membership):
        """Find the best split on X.
        Creates decision rules on each feature of X based on split values and
        given fuzziness ratio.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The array of input samples to be partitioned.
        y : array-like of shape (n_samples,)
            The array of labels.
        membership : array-like of shape (n_samples,)
            The membership of each sample.

        Returns
        -------
        (best_gain, best_rule) : tuple of (float, FuzzyDecisionRule or None)
            The best gain and decision rule that were found. If no split was
            found, equal to (0.0, None).
        """
        best_gain = 0.
        best_rule = None

        if np.unique(y).shape[0] <= 1:
            return best_rule, best_gain

        for feature_idx in range(X.shape[1]):
            sorted_ = np.sort(np.unique(X[:, feature_idx]))
            midpoint_splits = ((sorted_ + np.roll(sorted_, -1)) / 2)[:-1]

            for split_val in midpoint_splits:
                rule = FuzzyDecisionRule(sorted_, split_val, self.fuzziness, feature_idx)
                new_membership = rule.evaluate(X)
                gain = self.gain_function(y, membership, new_membership)
                if gain > best_gain:
                    best_rule, best_gain = rule, gain

        return best_rule, best_gain
