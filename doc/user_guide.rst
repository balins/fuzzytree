.. title:: User guide : contents

.. _user_guide:

==========
User guide
==========

The module
----------
The module contains a scikit-learn compatible of fuzzy decision tree estimator.
Currently, only single-output multiclass classification is supported.

Classifier
----------

Similarly to the classification tree known from scikit-learn, the
:class:`FuzzyDecisionTreeClassifier` implements ``fit``, ``predict``,
``predict_proba`` and ``predict_log_proba`` methods:

* at ``fit``, some parameters can be learned from ``X`` and ``y``. One can also
  provide weights for each sample in X. It will be treated as a fuzzy membership
  function on universe of samples;
* at ``predict``, predictions will be computed using ``X`` and the parameters
  learned during ``fit``. The output corresponds to the predicted class for each sample;
* ``predict_proba`` will give a 2D matrix where each column corresponds to the
  class and each entry will be the probability of the associated class.
* ``predict_log_proba`` will give a logarithm of ``predict_proba``.

:class:`FuzzyDecisionTreeClassifier` inherits from both
:class:`sklearn.base.BaseEstimator` and
:class:`sklearn.base.ClassifierMixin`, what enables it to be used with scikit-learn
objects, for example a pipeline::

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from fuzzytree import FuzzyDecisionTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> pipe = make_pipeline(StandardScaler(), FuzzyDecisionTreeClassifier())
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    Pipeline(...)

Then, you can call ``predict`` and ``predict_proba``::

    >>> pipe.predict(X)  # doctest: +ELLIPSIS
    array([...])
    >>> pipe.predict_proba(X)  # doctest: +ELLIPSIS
    array([...])

Since our classifier inherits from :class:`sklearn.base.ClassifierMixin`, we
can compute the accuracy by calling the ``score`` method::

    >>> pipe.score(X, y)  # doctest: +ELLIPSIS
    0...

Regressor
---------

As for now, regression is not supported :(