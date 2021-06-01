import numpy as np

# todo finish docs

def membership_ratio(y, membership, all_classes=None):
    if all_classes is None:
        all_classes = np.unique(y)

    membership_by_class = np.array([np.sum(membership[y == cls]) for cls in all_classes])

    return membership_by_class / membership_by_class.sum()


def split_by_membership(old_membership, new_membership):
    indices = np.flatnonzero(new_membership)
    old_and_new = old_membership[indices] * new_membership[indices]

    return old_and_new, indices


def s_shaped_membership(x, a, b):
    # from skfuzzy: https://scikit-fuzzy.github.io/scikit-fuzzy/_modules/skfuzzy/membership/generatemf.html#smf
    assert a <= b, 'a <= b is required.'
    y = np.ones(len(x))
    idx = x <= a
    y[idx] = 0

    idx = np.logical_and(a <= x, x <= (a + b) / 2.)
    y[idx] = 2. * ((x[idx] - a) / (b - a)) ** 2.

    idx = np.logical_and((a + b) / 2. <= x, x <= b)
    y[idx] = 1 - 2. * ((x[idx] - b) / (b - a)) ** 2.

    return y
