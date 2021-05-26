import warnings

import numpy as np


class DecisionNode:
    def __init__(self, rule, true_branch, false_branch, all_classes):
        self.rule = rule
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.classes_ = all_classes

    def predict_log_proba(self, X):
        mask = self.rule.evaluate(X)
        prediction = np.zeros((mask.shape[0], self.classes_.shape[0]))

        n_true = np.count_nonzero(mask)
        n_false = X.shape[0] - n_true

        if n_true > 0:
            prediction[mask] = self.true_branch.predict_log_proba(X[mask])
        if n_false > 0:
            prediction[~mask] = self.false_branch.predict_log_proba(X[~mask])

        return prediction


class Leaf:
    def __init__(self, y, all_classes):
        self.elements = y
        self.classes_ = all_classes
        self.counts = np.array([np.count_nonzero(y == cls) for cls in self.classes_])

    def predict_log_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classes_log_proba = np.log(self.counts) - np.log(self.elements.shape[0])

        prediction = np.array([classes_log_proba] * X.shape[0])
        return prediction

    def __contains__(self, item):
        return item in self.elements
