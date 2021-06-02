import pytest
from sklearn.datasets import load_iris

from fuzzytree import FuzzyDecisionTreeClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_template_classifier(data):
    X, y = data
    clf = FuzzyDecisionTreeClassifier(min_impurity_decrease=1)
    assert clf.min_impurity_decrease == 1

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
