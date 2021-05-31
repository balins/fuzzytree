import numpy as np


def split_by_membership(old_membership, new_membership):
    indices = np.flatnonzero(new_membership)
    old_and_new = old_membership[indices] * new_membership[indices]

    return old_and_new, indices
