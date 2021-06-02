import numpy as np

from ._utils import s_shaped_membership


class FuzzyDecisionRule:
    """Fuzzy decision rule that defines a split.

    Parameters
    ----------
    sorted_feature : array-like of shape (n_samples, n_features)
        The ascendingly sorted ndarray of input samples.
    split_val : float
        The value that defines a soft boundary of split
    fuzziness : float
        The fuzziness parameter that controls softness of boundary
        between 0. (hard) and 1. (soft).
    feature_idx : int
        The index of feature that provides the values for the split.

    Attributes
    ----------
    universe : ndarray of shape (n_classes,)
        The arguments for generated membership function.
    membership : array-like of shape (len(universe),)
        The array-like of membership values from each argument
        from universe.
    """

    def __init__(self, sorted_feature, split_val, fuzziness, feature_idx):
        self.split_val = split_val
        self.feature_idx = feature_idx

        min_, max_ = sorted_feature[0], sorted_feature[-1]
        margin_width = fuzziness * sorted_feature.std()
        a = split_val - margin_width
        b = split_val + margin_width

        self.universe = np.linspace(min_, max_, 300)
        self.membership = s_shaped_membership(self.universe, a, b)

    def evaluate(self, X):
        """Evaluate the decision rule on set of samples.
        Uses interpolation to get values for samples not belonging to self.universe.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The array of samples to be evaluated against the decision rule.

        Returns
        -------
        membership : array-like of shape (n_samples,)
            The array containing membership values of each sample. The more
            a sample fulfills the decision rule the bigger its membership
            value is.
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if n_features < self.feature_idx - 1:
            raise ValueError("input of this decision rule must have at least %s features (got %s)."
                             % (self.feature_idx + 1, n_features))
        if n_samples == 0:
            return np.array([])

        membership = self.evaluate_(X[:, self.feature_idx])

        return membership

    def evaluate_(self, x):
        return np.interp(x, self.universe, self.membership)

    def __str__(self):
        return "x[%s] >= %s?" % (self.feature_idx, self.split_val)

    def __repr__(self):
        return str(self)
