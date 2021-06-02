##########################
Quick Start with fuzzytree
##########################

This package provides a fuzzy decision tree algorithm implementation,
that is also scikit-learn compatible. Currently, only single-output
multiclass classification problems are supported.

Basic usage of `FuzzyDecisionTreeClassifier`
============================================

1. Load your dataset
-------------------------------------

>>> from sklearn.datasets import make_moons
>>> from sklearn.model_selection import train_test_split
>>> X, y = make_moons(n_samples=300, noise=0.5, random_state=42)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

2. Fit the classifier
---------------------

>>> from fuzzytree import FuzzyDecisionTreeClassifier
>>> clf_fuzz = FuzzyDecisionTreeClassifier().fit(X_train, y_train)

We will also make comparison to basic scikit-learn decision tree classifier

>>> from sklearn.tree import DecisionTreeClassifier
>>> clf_sk = DecisionTreeClassifier().fit(X_train, y_train)

3. Evaluate models on the test set
----------------------------------

>>> print(f"fuzzytree: {clf_fuzz.score(X_test, y_test)}")
>>> print(f"  sklearn: {clf_sk.score(X_test, y_test)}")

4. Plot the results
--------------------

We can also plot the results.

>>> from mlxtend.plotting import plot_decision_regions
>>> import matplotlib.pyplot as plt
>>> import matplotlib.gridspec as gridspec
>>> gs = gridspec.GridSpec(2, 2)
>>> fig = plt.figure(figsize=(10,8))
>>> labels = ['Fuzzy Decision Tree', 'sklearn Decision Tree']
>>> for clf, lab, grd in zip([clf_fuzz, clf_sk],
>>>                          labels, [[0, 0], [0, 1]]):
>>>     ax = plt.subplot(gs[grd[0], grd[1]])
>>>     fig = plot_decision_regions(X=X_train, y=y_train, clf=clf, legend=2)
>>>     plt.title("%s (train)" % lab)
>>> for clf, lab, grd in zip([clf_fuzz, clf_sk],
>>>                          labels, [[1, 0], [1, 1]]):
>>>     ax = plt.subplot(gs[grd[0], grd[1]])
>>>     fig = plot_decision_regions(X=X_test, y=y_test, clf=clf, legend=2)
>>>     plt.title("%s (test)" % lab)
>>> plt.show()

See the results in :ref:`general_examples`.