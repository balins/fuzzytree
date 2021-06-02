import numpy as np

from ._utils import split_by_membership, membership_ratio


def gini_index(y, membership, new_membership):
    """
    Calculate the Gini index for new membership values.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The membership of each label.
    new_membership : array-like of shape (n_samples,)
        The new membership of each label.

    Returns
    -------
    float : decrease of impurity measured by Gini index
    """

    gini_criterion_ = impurity_decrease(y, membership, new_membership, criterion=gini)

    return gini_criterion_


def entropy_decrease(y, membership, new_membership):
    """
    Calculate the entropy decrease for new membership values.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The membership of each label.
    new_membership : array-like of shape (n_samples,)
        The new membership of each label.

    Returns
    -------
    float : decrease of impurity measured by entropy
    """

    entropy_decrease_ = impurity_decrease(y, membership, new_membership, criterion=entropy)

    return entropy_decrease_


def misclassification_decrease(y, membership, new_membership):
    """
    Calculate the decrease in misclassification ratio for new membership values.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The membership of each label.
    new_membership : array-like of shape (n_samples,)
        The new membership of each label.

    Returns
    -------
    float : decrease of impurity measured by misclassification ratio
    """

    misclassification_ratio_ = impurity_decrease(y, membership, new_membership, criterion=misclassification)

    return misclassification_ratio_


def impurity_decrease(y, membership, new_membership, criterion):
    """
    A general function that calculates decrease in impurity.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        An array of labels.
    membership : array-like of shape (n_samples,)
        The membership of each label.
    new_membership : array-like of shape (n_samples,)
        The new membership of each label.
    criterion: callable
        The impurity function

    Returns
    -------
    float : decrease of impurity measured by given criterion
    """
    membership_true, indices_true = split_by_membership(membership, new_membership)
    membership_false, indices_false = split_by_membership(membership, 1. - new_membership)

    information_gain_ = criterion(y, membership) \
                        - membership_true.sum() / membership.sum() * criterion(y[indices_true], membership_true) \
                        - membership_false.sum() / membership.sum() * criterion(y[indices_false], membership_false)

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

    gini_ = 1. - np.sum(mr ** 2)

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

    entropy_ = -np.sum(mr * np.log(mr))

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

    misclassification_ = 1. - mr.max()

    return misclassification_
