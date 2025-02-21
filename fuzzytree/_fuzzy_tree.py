from collections import deque

import numpy as np

from ._utils import membership_ratio, joint_membership, weighted_mean


class FuzzyTreeBuilder:
    """A fuzzy decision tree builder.

       Parameters
       ----------
       splitter : Splitter
           The Splitter object, which provides node_split method to split an array-like
           of samples based on its feature values and membership of observations.
       max_depth : int
           The maximum depth of built tree.
       min_membership_split : float, default=2.0
           The minimum sum of membership required to split an internal node.
       min_membership_leaf : float, default=1.0
           The minimum sum of membership required to be at a leaf node.
           A split point at any depth will only be considered if it leaves at
           least ``min_membership_leaf`` sum of membership in each of the left and
           right branches. This may have the effect of smoothing the model,
           especially in regression.
       min_impurity_decrease : float
           The minimal gain in impurity decrease required for a node to be split.
       fuzzy_tree_class : class type
            Class for fuzzy tree class, could be FuzzyTree or FuzzyRegressionTree
    """

    def __init__(self,
                 splitter,
                 min_membership_split,
                 min_membership_leaf,
                 max_depth,
                 min_impurity_decrease,
                 fuzzy_tree_class):
        self.splitter = splitter
        self.min_membership_split = min_membership_split
        self.min_membership_leaf = min_membership_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.fuzzy_tree_class = fuzzy_tree_class

    def build(self, tree, X, y, membership):
        """Build a fuzzy decision tree.

        Parameters
        ----------
        tree : FuzzyTree
        X : array-like of shape (n_samples, n_features)
            The array of input samples to be partitioned.
        y : array-like of shape (n_samples,)
            The array of labels.
        membership : array-like of shape (n_samples,)
            The membership of each sample.
        """
        stack = deque()
        stack.append((tree, membership))

        while stack:
            tree, membership = stack.pop()

            unique_cls = np.unique(y[membership.nonzero()])
            if tree.level >= self.max_depth \
                    or unique_cls.shape[0] <= 1 or \
                    membership.sum() < self.min_membership_split:
                continue

            rule, membership_true, membership_false, gain = self.splitter.node_split(X, y, membership)
            if not rule or gain < self.min_impurity_decrease:
                continue

            tree.rule = rule
            tree.true_branch = self.fuzzy_tree_class(y, membership_true, tree.level + 1)
            tree.false_branch = self.fuzzy_tree_class(y, membership_false, tree.level + 1)

            stack.append((tree.true_branch, membership_true))
            stack.append((tree.false_branch, membership_false))


class FuzzyTree:
    """Fuzzy decision tree representation.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The array of labels.
    membership : array-like of shape (n_samples,)
        The membership of samples that respective labels are
        coming from.
    level : int
        Depth of the node.
    rule : FuzzyDecisionRule, default=None
        The rule that was used to split this node. If None,
        then the node is a leaf.
    true_branch : FuzzyTree, default=None
        The child node containing labels of samples which
        memberships of were evaluated as non-zero by the
        fuzzy decision rule. If None, then the node is a leaf.
    false_branch : FuzzyTree, default=None
        The child node containing labels of samples which
        memberships of were evaluated as non-zero by the
        inverse of the fuzzy decision rule. If None, then
        the node is a leaf.

    Attributes
    ----------
    class_weights : ndarray of shape (n_classes,)
        The membership ratio for each labels of n_classes.
    """

    def __init__(self, y, membership, level=0, rule=None, true_branch=None, false_branch=None):
        self.class_weights = membership_ratio(y, membership)
        self.level = level
        self.rule = rule
        self.true_branch = true_branch
        self.false_branch = false_branch

    def predict(self, X, membership):
        """Predict labels of each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples,n_features)
            The array of input samples.
        membership : array-like of shape (n_samples,)
            The array-like of membership of each sample.

        Returns
        -------
        y : array-like of shape (n_samples,n_classes)
            The probability of predicted classes, for each sample.
        """
        return (self._predict_leaf(membership) if self.is_leaf
                else self._predict_internal(X, membership))

    def _predict_internal(self, X, membership):
        new_membership = self.rule.evaluate(X)

        membership_true = joint_membership(membership, new_membership)
        membership_false = joint_membership(membership, 1 - new_membership)

        prediction = np.zeros((X.shape[0], self.class_weights.shape[0]))

        if membership_true.any():
            prediction = self.true_branch.predict(X, membership_true)
        if membership_false.any():
            prediction += self.false_branch.predict(X, membership_false)

        return prediction

    def _predict_leaf(self, membership):
        return joint_membership(membership[:, None], self.class_weights)

    @property
    def max_depth(self):
        """Return the depth of the decision tree.

        Returns
        -------
        int : The maximum depth of the tree.
        """
        return (self.level if self.is_leaf
                else max(self.true_branch.max_depth, self.false_branch.max_depth))

    @property
    def n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        int : The number of leaves.
        """
        return (1 if self.is_leaf
                else self.true_branch.n_leaves + self.false_branch.n_leaves)

    @property
    def is_leaf(self):
        """Return whether the node is a leaf.

        Returns
        -------
        bool : True if node is a leaf.
        """
        return self.rule is None


class FuzzyRegressionTree(FuzzyTree):
    """Fuzzy decision tree for regression.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The array of y values.
        membership : array-like of shape (n_samples,)
            The membership of samples that respective labels are
            coming from.
        level : int
            Depth of the node.
        rule : FuzzyDecisionRule, default=None
            The rule that was used to split this node. If None,
            then the node is a leaf.
        true_branch : FuzzyTree, default=None
            The child node containing labels of samples which
            memberships of were evaluated as non-zero by the
            fuzzy decision rule. If None, then the node is a leaf.
        false_branch : FuzzyTree, default=None
            The child node containing labels of samples which
            memberships of were evaluated as non-zero by the
            inverse of the fuzzy decision rule. If None, then
            the node is a leaf.

        Attributes
        ----------
        weighted_mean : float
            The weighted_mean of y values on the node
        """
    def __init__(self, y, membership, level=0, rule=None, true_branch=None, false_branch=None):
        self.weighted_mean = weighted_mean(y, membership)
        self.level = level
        self.rule = rule
        self.true_branch = true_branch
        self.false_branch = false_branch

    def _predict_internal(self, X, membership):
        new_membership = self.rule.evaluate(X)

        membership_true = joint_membership(membership, new_membership)
        membership_false = joint_membership(membership, 1 - new_membership)

        prediction = np.zeros((X.shape[0], 1))

        if membership_true.any():
            prediction = self.true_branch.predict(X, membership_true)
        if membership_false.any():
            prediction += self.false_branch.predict(X, membership_false)

        return prediction

    def _predict_leaf(self, membership):
        return joint_membership(membership[:, None], self.weighted_mean)