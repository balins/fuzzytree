import numpy as np


def membership_ratio(y, membership, all_classes=None):
    """Calculate the membership ratio of each class in y.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The membership of each sample that corresponding label is
        coming from.
    all_classes : array-like of shape (n_classes,) or None, default=None
        The array of all possible labels, including ones that may not
        be present in y.

    Returns
    -------
    membership_by_class : array-like of float of shape (max(len(all_classes), len(np.unique(y)),)
        The membership ratio for each class in np.unique(y) or all_classes if provided.
    """
    if all_classes is None:
        all_classes = np.unique(y)

    membership_by_class = np.array([np.sum(membership[y == cls]) for cls in all_classes])
    membership_by_class /= membership_by_class.sum()

    return membership_by_class


def split_by_membership(old_membership, new_membership):
    """Calculate the new membership ratio.

    Parameters
    ----------
    old_membership : array-like of shape (n_samples,)
        The array of initial membership of each sample.
    new_membership : array-like of shape (n_samples,)
        The new membership of each sample, calculated by some fuzzy decision rule.

    Returns
    -------
    (old_and_new, indices) : tuple of (array-like, array-like)
        old_and_new refers to an array-like of shape (n_samples - len(np.where(new_membership == 0)),)
        and contains product of new and old membership arrays. indices is an array-like of shape
        (n_samples - len(np.where(new_membership == 0)),) and contains indices of non-zero memberships
        in new_membership
    """
    indices = np.flatnonzero(new_membership)
    old_and_new = old_membership[indices] * new_membership[indices]

    return old_and_new, indices


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
    s_func = np.ones(len(universe))
    idx = universe <= a
    s_func[idx] = 0

    idx = np.logical_and(a <= universe, universe <= (a + b) / 2.)
    s_func[idx] = 2. * ((universe[idx] - a) / (b - a)) ** 2.

    idx = np.logical_and((a + b) / 2. <= universe, universe <= b)
    s_func[idx] = 1 - 2. * ((universe[idx] - b) / (b - a)) ** 2.

    return s_func
