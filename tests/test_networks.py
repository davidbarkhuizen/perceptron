import random

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.axes import Axes

from perceptron.graphics.chart import new_axes, new_figure, plot_linear_classifier_network, plot_training_data
from perceptron.model.linear_classifier_network import LinearClassifierNetwork
from perceptron.train import (
    classification_disagreement_rate,
    random_alternating_training_data,
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


def test_classification_disagreement_rate_is_zero_for_identical_classifier():

    classifier = LinearClassifierNetwork(2, 2, [(-5.0, 5.0), (-5.0, 5.0)])
    classifier.randomize()

    assert classification_disagreement_rate(classifier, classifier) == 0.0


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

    disagreement_before = classification_disagreement_rate(reference, student, sample_count=1000)
    train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=5)
    disagreement_after = classification_disagreement_rate(reference, student, sample_count=1000)

    assert disagreement_after < disagreement_before
    assert disagreement_after < 0.15
