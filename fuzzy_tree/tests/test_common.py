import pytest
from sklearn.utils.estimator_checks import check_classifiers_classes, check_classifiers_train, \
    check_classifiers_predictions, check_classifiers_one_label, check_classifier_multioutput, \
    check_classifier_data_not_an_array, check_classifiers_regression_target, \
    check_classifiers_multilabel_representation_invariance

from fuzzy_tree import FuzzyDecisionTreeClassifier


@pytest.mark.parametrize(
    "test", [check_classifiers_classes, check_classifiers_train, check_classifiers_predictions,
             check_classifiers_one_label, check_classifier_multioutput, check_classifier_data_not_an_array,
             check_classifiers_regression_target, check_classifiers_multilabel_representation_invariance]
)
def test_classifier(test):
    test("fdt", FuzzyDecisionTreeClassifier)
