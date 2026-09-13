from perceptron.geometry import is_positive_region_bounded, square_bounds
from perceptron.model.linear_classifier_network import LinearClassifierNetwork

from helpers import classifier_with_bounded_square_region


def test_square_bounds():

    assert square_bounds(10.0) == [(-10.0, 10.0), (-10.0, 10.0)]
    assert square_bounds(10.0, dimension=3) == [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]


def test_is_positive_region_bounded_true_for_a_known_bounded_square():

    bounds = square_bounds(10.0)
    classifier = classifier_with_bounded_square_region(bounds)

    assert is_positive_region_bounded(classifier) is True


def test_is_positive_region_bounded_false_for_cardinality_one():

    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork.randomized(1, 2, bounds)

    # a single half-plane can never be a bounded region
    assert is_positive_region_bounded(classifier) is False
