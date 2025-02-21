import numpy as np

from ._utils import membership_ratio


def gini_index(y, membership, membership_true, membership_false):
    """
    Calculate the Gini index for new membership values.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The old membership of each label.
    membership_true : array-like of shape (n_samples,)
        The new membership of each label.
    membership_false : array-like of shape (n_samples,)
        The complement of new membership of each label.

    Returns
    -------
    float : decrease of impurity measured by Gini index
    """

    gini_criterion_ = impurity_decrease(y, membership, membership_true, membership_false, gini)

    return gini_criterion_


def entropy_decrease(y, membership, membership_true, membership_false):
    """
    Calculate the entropy decrease for new membership values.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The old membership of each label.
    membership_true : array-like of shape (n_samples,)
        The new membership of each label.
    membership_false : array-like of shape (n_samples,)
        The complement of new membership of each label.

    Returns
    -------
    float : decrease of impurity measured by entropy
    """

    entropy_decrease_ = impurity_decrease(y, membership, membership_true, membership_false, entropy)

    return entropy_decrease_


def misclassification_decrease(y, membership, membership_true, membership_false):
    """
    Calculate the decrease in misclassification ratio for new membership values.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The old membership of each label.
    membership_true : array-like of shape (n_samples,)
        The new membership of each label.
    membership_false : array-like of shape (n_samples,)
        The complement of new membership of each label.

    Returns
    -------
    float : decrease of impurity measured by misclassification ratio
    """

    misclassification_ratio_ = impurity_decrease(y, membership, membership_true, membership_false, misclassification)

    return misclassification_ratio_


def variance_decrease(y, membership, membership_true, membership_false):
    """
        Calculate the decrease in variance for new membership values.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The array of y values which could be float type.
        membership : array-like of shape (n_samples,)
            The old membership of each label.
        membership_true : array-like of shape (n_samples,)
            The new membership of each label.
        membership_false : array-like of shape (n_samples,)
            The complement of new membership of each label.

        Returns
        -------
        float : decrease of impurity measured by misclassification ratio
    """
    var_decrease = impurity_decrease(y, membership, membership_true, membership_false, regression_variance)

    return var_decrease

def impurity_decrease(y, membership, membership_true, membership_false, criterion):
    """
    A general function that calculates decrease in impurity.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        An array of labels.
    membership : array-like of shape (n_samples,)
        The old membership of each label.
    membership_true : array-like of shape (n_samples,)
        The new membership of each label.
    membership_false : array-like of shape (n_samples,)
        The complement of new membership of each label.
    criterion: callable
        The impurity function

    Returns
    -------
    float : decrease of impurity measured by given criterion
    """
    information_gain_ = criterion(y, membership) \
                        - (membership_true.sum() / membership.sum()) * criterion(y, membership_true) \
                        - (membership_false.sum() / membership.sum()) * criterion(y, membership_false)

    return information_gain_


def gini(y, membership):
    """
    Calculates decrease in impurity by Gini criterion.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The membership of each label.

    Returns
    -------
    float : impurity measured by Gini criterion
    """
    mr = membership_ratio(y, membership)

    gini_ = 1 - np.sum(mr ** 2)

    return gini_


def entropy(y, membership):
    """
    Calculates decrease in impurity by entropy.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The membership of each label.

    Returns
    -------
    float : impurity measured by entropy
    """
    mr = membership_ratio(y, membership)
    mr_nonzero = mr[mr.nonzero()]

    entropy_ = -np.sum(mr_nonzero * np.log(mr_nonzero))

    return entropy_


def misclassification(y, membership):
    """
    Calculates decrease in impurity by misclassification ratio.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The membership of each label.

    Returns
    -------
    float : impurity measured by misclassification ratio
    """
    mr = membership_ratio(y, membership)

    misclassification_ = 1 - mr.max()

    return misclassification_

def regression_variance(y, membership):
    """
    Calculates weighted variance for regression impurity.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of target values (continuous).
    membership : array-like of shape (n_samples,)
        The membership of each target value.

    Returns
    -------
    float : weighted variance.
    """
    # Normalize memberships to use as weights
    total_membership = np.sum(membership)
    if total_membership == 0:
        return 0.0

    weights = membership / total_membership

    weighted_mean = np.sum(weights * y)
    variance_ = np.sum(weights * (y - weighted_mean) ** 2)

    return variance_

