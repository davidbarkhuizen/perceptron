import random

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.axes import Axes

from perceptron.graphics.chart import (
    new_axes,
    new_figure,
    plot_linear_classifier_network,
    plot_training_data,
    reference_region_bounds,
)
from perceptron.model.linear_classifier_network import LinearClassifierNetwork
from perceptron.train import (
    class_balanced_disagreement_rate,
    random_alternating_training_data,
    reachable_reference_and_training_data,
    smoothed_series,
    train_linear_classifier_network,
)


def test_generation_of_random_test_data_from_reference_classifier():

    classifier_cardinality = 1
    dimension: int = 2
    l: float = 7.0
    training_set_size: int = 50

    x_min, x_max = -l, l
    y_min, y_max = -l, l
    input_bounds = [(x_min, x_max), (y_min, y_max)]

    classifier = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    classifier.randomize()

    training_data = random_alternating_training_data(training_set_size, classifier)

    assert len(training_data) == training_set_size


def test_generation_of_random_test_data_for_non_2d_classifier():

    dimension: int = 3
    training_set_size: int = 50
    input_bounds = [(-7.0, 7.0)] * dimension

    classifier = LinearClassifierNetwork(1, dimension, input_bounds)
    classifier.randomize()

    training_data = random_alternating_training_data(training_set_size, classifier)

    assert len(training_data) == training_set_size
    assert all(len(state) == dimension for state, _ in training_data)


def test_training_of_linear_classifier():

    classifier_cardinality: int = 1
    dimension: int = 2
    l: float = 10.0

    learning_rate: float = 0.25
    training_set_size: int = 1000
    epoch_count: int = 1

    x_min, x_max = -l, l
    y_min, y_max = -l, l
    input_bounds = [(x_min, x_max), (y_min, y_max)]

    # generate a (random) reference classifier network
    #
    reference_classifier = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    reference_classifier.randomize()

    # use the reference classifier to produce a set of training data
    #
    training_data = random_alternating_training_data(training_set_size, reference_classifier)

    # generate a new random classifier network for training
    #
    student_classifier = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    student_classifier.randomize()

    convergence_series: list[tuple[int, float]] = train_linear_classifier_network(
        student_classifier,
        training_data,
        learning_rate=learning_rate,
        epochs=epoch_count,
        reference_classifier=reference_classifier,
    )

    convergence_figure = new_figure("convergence")

    convergence_bounds = [
        (0.0, float(training_set_size)),
        (0.0, 1.0),
    ]

    convergence_axes: Axes = new_axes(convergence_figure, convergence_bounds, scaled=False)

    n: list[int] = [x[0] for x in convergence_series]
    disagreement: list[float] = [x[1] for x in convergence_series]
    convergence_axes.plot(n, disagreement)  # , '.', color='yellow')

    # ---------------------

    training_data_figure = new_figure("perceptrons (reference, student) with training data")
    axes = new_axes(training_data_figure, student_classifier.input_bounds)

    axes.set_xlim(input_bounds[0])
    axes.set_ylim(input_bounds[1])

    plot_training_data(axes, training_data)
    plot_linear_classifier_network(axes, reference_classifier, color="green")
    plot_linear_classifier_network(axes, student_classifier, color="purple")

    pyplot.close("all")


def test_class_balanced_disagreement_rate_is_zero_for_identical_classifier():

    classifier, _ = reachable_reference_and_training_data(2, 2, [(-5.0, 5.0), (-5.0, 5.0)], 10)

    assert class_balanced_disagreement_rate(classifier, classifier) == 0.0


def test_class_balanced_disagreement_rate_detects_error_the_old_metric_missed():

    random.seed(1)

    cardinality, dimension, l = 4, 2, 10.0
    bounds = [(-l, l), (-l, l)]

    reference = LinearClassifierNetwork(cardinality, dimension, bounds)
    reference.randomize()
    student = LinearClassifierNetwork(cardinality, dimension, bounds)
    student.randomize()

    assert class_balanced_disagreement_rate(reference, student, per_class_sample_count=200) > 0.3


def test_reachable_reference_and_training_data_returns_class_balanced_data():

    cardinality = 4
    dimension = 2
    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    training_set_size = 200

    reference, training_data = reachable_reference_and_training_data(cardinality, dimension, bounds, training_set_size)

    assert reference.cardinality == cardinality
    assert len(training_data) == training_set_size
    assert all(len(state) == dimension for state, _ in training_data)


def test_reference_region_bounds_expands_to_include_a_bounded_region():

    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    classifier = LinearClassifierNetwork(4, 2, bounds)

    # positive region is exactly the square [-1, 1] x [-1, 1]:
    # x > -1, x < 1, y > -1, y < 1
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [([1.0, 0.0], 1.0), ([-1.0, 0.0], 1.0), ([0.0, 1.0], 1.0), ([0.0, -1.0], 1.0)],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold

    small_fallback = [(-0.5, 0.5), (-0.5, 0.5)]
    (x_min, x_max), (y_min, y_max) = reference_region_bounds(classifier, small_fallback)

    assert x_min < -1.0 and x_max > 1.0
    assert y_min < -1.0 and y_max > 1.0


def test_reference_region_bounds_leaves_fallback_unchanged_when_unbounded():

    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    classifier = LinearClassifierNetwork(1, 2, bounds)
    classifier.randomize()

    assert reference_region_bounds(classifier, bounds) == bounds


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


def test_learn_reduces_to_single_node_update_for_cardinality_one():

    dimension = 2
    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    learning_rate = 0.25

    for category in (0, 1):
        network = LinearClassifierNetwork(1, dimension, bounds)
        network.randomize()
        node = network.hidden_layer.nodes[0]
        weights_before = list(node.input_node_weights)
        threshold_before = node.threshold

        state = (3.0, -4.0)
        network.learn(learning_rate, state, category)

        expected_network = LinearClassifierNetwork(1, dimension, bounds)
        expected_node = expected_network.hidden_layer.nodes[0]
        expected_node.update_input_weights(weights_before)
        expected_node.threshold = threshold_before
        expected_network.update_state_layer(state)
        expected_node.learn(learning_rate, category)

        assert node.input_node_weights == expected_node.input_node_weights
        assert node.threshold == expected_node.threshold


def test_training_of_cardinality_two_linear_classifier_reduces_disagreement():

    random.seed(0)

    cardinality: int = 2
    dimension: int = 2
    l: float = 10.0
    bounds = [(-l, l), (-l, l)]

    reference = LinearClassifierNetwork(cardinality, dimension, bounds)
    reference.randomize()
    training_data = random_alternating_training_data(400, reference)

    student = LinearClassifierNetwork(cardinality, dimension, bounds)
    student.randomize()

    disagreement_before = class_balanced_disagreement_rate(reference, student, per_class_sample_count=500)
    train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=5)
    disagreement_after = class_balanced_disagreement_rate(reference, student, per_class_sample_count=500)

    assert disagreement_after < disagreement_before
    assert disagreement_after < 0.15


def test_cardinality_must_be_at_least_one():

    with pytest.raises(AssertionError):
        LinearClassifierNetwork(0, 2, [(-1.0, 1.0), (-1.0, 1.0)])
