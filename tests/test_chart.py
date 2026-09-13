import matplotlib

matplotlib.use("Agg")

from perceptron.geometry import square_bounds
from perceptron.graphics.chart import disagreement_axis_bounds, reference_region_bounds
from perceptron.model.linear_classifier_network import LinearClassifierNetwork

from helpers import classifier_with_bounded_square_region


def test_reference_region_bounds_expands_to_include_a_bounded_region():

    bounds = square_bounds(10.0)
    classifier = classifier_with_bounded_square_region(bounds)

    small_fallback = [(-0.5, 0.5), (-0.5, 0.5)]
    (x_min, x_max), (y_min, y_max) = reference_region_bounds(classifier, small_fallback)

    assert x_min < -1.0 and x_max > 1.0
    assert y_min < -1.0 and y_max > 1.0


def test_reference_region_bounds_leaves_fallback_unchanged_when_unbounded():

    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork.randomized(1, 2, bounds)

    assert reference_region_bounds(classifier, bounds) == bounds


def test_disagreement_axis_bounds():

    assert disagreement_axis_bounds(500.0) == [(0.0, 500.0), (0.0, 1.0)]
    assert disagreement_axis_bounds(500.0, log=True) == [(0.0, 500.0), (1.0e-3, 1.0)]
