import numpy as np

from fuzzy_tree._utils import split_by_membership


def membership_ratio(y, membership):
    unique = np.unique(y)
    membership_by_class = np.array([np.sum(membership[y == cls]) for cls in unique])

    return membership_by_class / membership_by_class.sum()


def gini(y, membership):
    """
    Given an array of labels, calculate its Gini impurity
    y: array of labels
    """
    mr = membership_ratio(y, membership)

    gini_ = 1. - np.sum(mr ** 2.)

    return gini_


def entropy(y, membership):
    """
    Given an array-like, calculate its entropy.
    y: NumPy array-like of labels
    """
    mr = membership_ratio(y, membership)

    if not 0.999 < mr.sum() < 1.001:
        raise ValueError(mr.sum())

    entropy_ = -np.sum(mr * np.log(mr))

    if not 0.0 <= entropy_ < 1.1:
        raise ValueError(entropy_)

    return min(entropy_, 1.)


def misclassification(y, membership):
    """
    Given an array-like, calculate its entropy.
    y: NumPy array-like of labels
    """
    mr = membership_ratio(y, membership)

    impurity = 1. - mr.max()

    return impurity


def impurity_decrease(y, membership, new_membership, criterion):
    """
    Given a NumPy array-like and its split masks, calculate the information gain of that split
    y: a NumPy array-like of split feature values
    masks: split choices
    """
    membership_true, indices_true = split_by_membership(membership, new_membership)
    membership_false, indices_false = split_by_membership(membership, 1. - new_membership)

    if not abs(membership.sum() - (membership_true.sum() + membership_false.sum())) < 5:
        print(membership_true.sum() + membership_false.sum())
        print(membership)
        print(new_membership)
        print(membership_true)
        print(membership_false)
        raise ValueError(membership.sum())

    information_gain_ = criterion(y, membership) \
                        - membership_true.sum() / membership.sum() * criterion(y[indices_true], membership_true) \
                        - membership_false.sum() / membership.sum() * criterion(y[indices_false], membership_false)

    return information_gain_


def gini_criterion(y, membership, new_membership):
    """
    Given a NumPy array-like and its split masks, calculate the information gain ratio of that split
    y: a NumPy array-like of split feature values
    mask: split choices
    """

    gini_criterion_ = impurity_decrease(y, membership, new_membership, gini)

    return gini_criterion_


def gain_ratio(y, membership, new_membership):
    """
    Given a NumPy array-like and its split masks, calculate the information gain ratio of that split
    y: a NumPy array-like of split feature values
    mask: split choices
    """

    gain_ratio_ = impurity_decrease(y, membership, new_membership, entropy) / entropy(y, membership)

    return gain_ratio_


def misclassification_ratio(y, membership, new_membership):
    """
    Given a NumPy array-like and its split masks, calculate the information gain ratio of that split
    y: a NumPy array-like of split feature values
    mask: split choices
    """

    misclassification_ratio_ = impurity_decrease(y, membership, new_membership, misclassification)

    return misclassification_ratio_
