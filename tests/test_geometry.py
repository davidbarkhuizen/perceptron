from perceptron.geometry import is_positive_region_bounded, reference_positive_region_polygon, square_bounds
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


def test_is_positive_region_bounded_false_for_a_genuinely_empty_region():

    # the cardinality=1 case above is unbounded but non-empty (half of the plane still
    # satisfies it). Contradictory half-planes (x > 5 and x < -5) are a distinct scenario -
    # a genuinely infeasible, empty intersection - previously unexercised by any test.
    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork(2, 2, bounds)
    greater_than_five, less_than_negative_five = classifier.hidden_layer.nodes
    greater_than_five.update_input_weights([1.0, 0.0])
    greater_than_five.threshold = -5.0
    less_than_negative_five.update_input_weights([-1.0, 0.0])
    less_than_negative_five.threshold = -5.0

    assert reference_positive_region_polygon(classifier) == []
    assert is_positive_region_bounded(classifier) is False
