import random

import pytest

from perceptron.evaluate import class_balanced_disagreement_rate, compare_on_random_point, smoothed_series
from perceptron.geometry import square_bounds
from perceptron.model.linear_classifier_network import LinearClassifierNetwork
from perceptron.train import reachable_reference_and_training_data


def test_class_balanced_disagreement_rate_is_zero_for_identical_classifier():

    classifier, _ = reachable_reference_and_training_data(2, 2, square_bounds(5.0), 10)

    assert class_balanced_disagreement_rate(classifier, classifier) == 0.0


def test_class_balanced_disagreement_rate_detects_error_the_old_metric_missed():

    random.seed(1)

    cardinality, dimension, l = 4, 2, 10.0
    bounds = square_bounds(l, dimension)

    reference = LinearClassifierNetwork.randomized(cardinality, dimension, bounds)
    student = LinearClassifierNetwork.randomized(cardinality, dimension, bounds)

    assert class_balanced_disagreement_rate(reference, student, per_class_sample_count=200) > 0.3


def test_class_balanced_disagreement_rate_raises_for_an_unreachable_reference_class():

    bounds = square_bounds(10.0)
    unreachable = LinearClassifierNetwork(1, 2, bounds)
    node = unreachable.hidden_layer.nodes[0]
    # tiny weights + a large threshold mean the decision boundary never crosses these
    # bounds, so one class can never be sampled
    node.update_input_weights([0.01, 0.01])
    node.threshold = -5.0

    with pytest.raises(RuntimeError):
        class_balanced_disagreement_rate(unreachable, unreachable, per_class_sample_count=10, max_attempts=200)


def test_compare_on_random_point_classifies_with_both_networks():

    classifier, _ = reachable_reference_and_training_data(2, 2, square_bounds(5.0), 10)

    state, reference_category, student_category = compare_on_random_point(classifier, classifier)

    assert len(state) == 2
    assert all(bound[0] <= value <= bound[1] for value, bound in zip(state, classifier.input_bounds))
    assert reference_category == student_category == classifier.classify_state(state)


def test_smoothed_series_with_window_one_is_identity():

    values = [0.0, 1.0, 0.0, 1.0]

    assert smoothed_series(values, window=1) == values


def test_smoothed_series_is_a_trailing_moving_average():

    values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    smoothed = smoothed_series(values, window=3)

    assert len(smoothed) == len(values)
    assert smoothed[0] == 0.0
    assert smoothed[1] == 0.5
    assert smoothed[2] == pytest.approx(1 / 3)
    assert smoothed[3] == pytest.approx(2 / 3)
