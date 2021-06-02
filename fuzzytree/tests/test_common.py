import pytest
from sklearn.utils.estimator_checks import check_estimator

from fuzzytree import FuzzyDecisionTreeClassifier


@pytest.mark.parametrize(
    "Estimator", [FuzzyDecisionTreeClassifier()]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
