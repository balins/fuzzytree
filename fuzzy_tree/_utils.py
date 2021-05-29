import numpy as np


def powerset(x):
    def bits_array(num):
        return np.unpackbits(np.array(num, dtype=np.uint8), bitorder='little')[:len(unique)]

    unique = np.unique(x)
    bits = 2 ** len(unique)

    if bits > 8:
        raise NotImplementedError("Powersets for more than 8 unique values are "
                                  "not currently supported. Consider one-hot-encoding for categorical "
                                  "features with more than 8 unique values")

    return np.array([unique[np.logical_and(unique, bits_array(i))] for i in range(bits)])[1:-1]


def midpoints(x):
    sorted_ = np.sort(x)
    return ((sorted_ + np.roll(sorted_, -1)) / 2.)[:-1]
