import numpy as np

from ._fuzzy_decision_rule import FuzzyDecisionRule
from ._utils import joint_membership


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
        self._sorted_cols = None

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
        (best_rule, best_true, best_false, best_gain) : tuple of (FuzzyDecisionRule, array-like, array_like, float)
            The best decision rule, pair of new memberships generated by this rule and the best gain that was found.
            If no split was found, equal to (None, None, None, 0).
        """
        if self._sorted_cols is None:
            self._init_sorted_cols(X)

        best_rule, best_true, best_false, best_gain = None, None, None, 0
        nonzero = membership.nonzero()

        for feature_idx in range(X.shape[1]):
            sorted_unique = np.unique(self._sorted_cols[feature_idx][nonzero])
            midpoint_splits = ((sorted_unique + np.roll(sorted_unique, -1)) / 2)[:-1]

            for split_val in midpoint_splits:
                rule = FuzzyDecisionRule(sorted_unique, split_val, self.fuzziness, feature_idx)

                new_membership = rule.evaluate(X)
                membership_true = joint_membership(membership, new_membership)
                membership_false = joint_membership(membership, 1 - new_membership)
                gain = self.gain_function(y, membership, membership_true, membership_false)

                if gain > best_gain:
                    best_rule, best_true, best_false, best_gain = rule, membership_true, membership_false, gain

        return best_rule, best_true, best_false, best_gain

    def _init_sorted_cols(self, X):
        self._sorted_cols = {}

        for feature_idx in range(X.shape[1]):
            self._sorted_cols[feature_idx] = np.sort(X[:, feature_idx])
