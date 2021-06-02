"""
=========================================================
Comparison of crisp and fuzzy classifiers on iris dataset
=========================================================

A comparison plot for :class:`FuzzyDecisionTreeClassifier`
and sklearn's :class:`DecisionTreeClassifier` on iris
dataset (only two features were selected)
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from fuzzytree import FuzzyDecisionTreeClassifier

iris = load_iris()

features = [2, 3]

X = iris.data[:, features]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf_fuzz = FuzzyDecisionTreeClassifier().fit(X_train, y_train)
clf_sk = DecisionTreeClassifier().fit(X_train, y_train)

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))
labels = ['Fuzzy Decision Tree', 'sklearn Decision Tree']
for clf, lab, grd in zip([clf_fuzz, clf_sk],
                         labels, [[0, 0], [0, 1]]):
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_train, y=y_train, clf=clf, legend=2)
    plt.xlabel(iris.feature_names[features[0]])
    plt.ylabel(iris.feature_names[features[1]])
    plt.title("%s (train)" % lab)
for clf, lab, grd in zip([clf_fuzz, clf_sk],
                         labels, [[1, 0], [1, 1]]):
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_test, y=y_test, clf=clf, legend=2)
    plt.xlabel(iris.feature_names[features[0]])
    plt.ylabel(iris.feature_names[features[1]])
    plt.title("%s (test)" % lab)

plt.show()
