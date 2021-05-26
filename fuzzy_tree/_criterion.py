import numpy as np


def frequency_ratio(y):
    _, counts = np.unique(y, return_counts=True)
    return counts / y.shape[0]


def gini(y):
    """
    Given an array of labels, calculate its Gini impurity
    y: array of labels
    """
    freq_ratio = frequency_ratio(y)

    gini_ = 1 - np.sum(freq_ratio ** 2)

    return gini_


def entropy(y):
    """
    Given an array-like, calculate its entropy.
    y: NumPy array-like of labels
    """
    freq_ratio = frequency_ratio(y)

    entropy_ = -np.sum(freq_ratio * np.log2(freq_ratio))

    return entropy_


def misclassification(y):
    """
    Given an array-like, calculate its entropy.
    y: NumPy array-like of labels
    """
    if y.shape[0] == 0:
        return 0

    labels, _ = np.unique(y, return_counts=True)

    labels_freq = labels / y.shape[0]

    impurity = 1 - labels_freq.max()

    return impurity


def impurity_decrease(X, y, mask, criterion):
    """
    Given a NumPy array-like and its split masks, calculate the information gain of that split
    y: a NumPy array-like of split feature values
    masks: split choices
    """
    n_samples = X.shape[0]

    n_true = np.count_nonzero(mask)
    n_false = n_samples - n_true

    information_gain_ = criterion(y) \
                        - n_true / n_samples * criterion(y[mask]) \
                        - n_false / n_samples * criterion(y[~mask])

    return information_gain_


def gini_criterion(X, y, mask):
    """
    Given a NumPy array-like and its split masks, calculate the information gain ratio of that split
    y: a NumPy array-like of split feature values
    mask: split choices
    """

    gini_criterion_ = impurity_decrease(X, y, mask, gini)

    return gini_criterion_


def gain_ratio(X, y, mask):
    """
    Given a NumPy array-like and its split masks, calculate the information gain ratio of that split
    y: a NumPy array-like of split feature values
    mask: split choices
    """

    gain_ratio_ = impurity_decrease(X, y, mask, entropy) / entropy(y)

    return gain_ratio_


def misclassification_ratio(X, y, mask):
    """
    Given a NumPy array-like and its split masks, calculate the information gain ratio of that split
    y: a NumPy array-like of split feature values
    mask: split choices
    """

    misclassification_ratio_ = impurity_decrease(X, y, mask, misclassification)

    return misclassification_ratio_
