import numpy as np


def membership_ratio(y, membership):
    """Calculate the membership ratio of each class in y.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The membership of each sample that corresponding label is
        coming from.

    Returns
    -------
    membership_by_class : array-like of float of shape (len(np.unique(y)),)
        The membership ratio for each class in np.unique(y).
    """

    membership_by_class = np.bincount(y, weights=membership)
    if membership_by_class.any():
        membership_by_class /= membership_by_class.sum()

    return membership_by_class

def weighted_mean(y, membership):
    """
    Calculate the weighted mean of the target values.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of continuous target values.
    membership : array-like of shape (n_samples,)
        The membership of each sample.

    Returns
    -------
    weighted_mean : float
        The weighted mean of the target values.
    """
    if membership.sum() == 0:
        raise ValueError("Total membership sum cannot be zero.")

    weighted_mean = np.sum(y * membership) / np.sum(membership)
    return weighted_mean

def joint_membership(a, b):
    """Calculate the joint membership of arrays.

    Parameters
    ----------
    a : array-like of shape (n_samples,)
        The first array of membership.
    b : array-like of shape (n_samples,)
        The second array of membership.

    Returns
    -------
    joint_membership_ : array-like of float of shape (n_samples,)
        The joint membership of a and b.
    """
    joint_membership_ = a * b

    return joint_membership_


def s_shaped_membership(universe, a, b):
    """Calculate the s-shaped membership function.

    Parameters
    ----------
    universe : array-like of shape (n_elements,)
        The array containing arguments to s-shaped function.
    a : float
        The argument where the s-shaped function starts rising from 0.
    b : float
        The argument where the s-shaped function reaches 1 and stops rising.

    Returns
    -------
    s_func : array-like of shape (n_elements,)
        An array-like of floats containing values of s-shaped function given
        arguments from x.

    References
    ----------
    This function's implementation is taken from [1]_.
    .. [1] J. Warner, scikit-fuzzy Python module.
       See https://scikit-fuzzy.github.io/scikit-fuzzy/_modules/skfuzzy/membership/generatemf.html#smf
    """
    s_func = np.ones_like(universe)
    idx = universe <= a
    s_func[idx] = 0

    idx = np.logical_and(a <= universe, universe <= (a + b) / 2)
    s_func[idx] = 2 * ((universe[idx] - a) / (b - a)) ** 2

    idx = np.logical_and((a + b) / 2 <= universe, universe <= b)
    s_func[idx] = 1 - 2 * ((universe[idx] - b) / (b - a)) ** 2

    return s_func
