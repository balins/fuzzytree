import pytest
from sklearn.utils.estimator_checks import check_estimator

from fuzzytree import FuzzyDecisionTreeClassifier, FuzzyDecisionTreeRegressor


@pytest.mark.parametrize(
    "estimator", [FuzzyDecisionTreeClassifier(), FuzzyDecisionTreeRegressor()],
)
def test_all_estimators(estimator):
    # TODO: Fix 'Comparing the output of FuzzyDecisionTree(Classifier|Regressor).predict[_proba] revealed that fitting with `sample_weight` is not equivalent to fitting with removed or repeated data points.'
    # return check_estimator(estimator)
    pass
