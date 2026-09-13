import pytest

from perceptron.geometry import square_bounds
from perceptron.model.linear_classifier_network import LinearClassifierNetwork


def test_cardinality_must_be_at_least_one():

    with pytest.raises(AssertionError):
        LinearClassifierNetwork(0, 2, [(-1.0, 1.0), (-1.0, 1.0)])


def test_randomized_returns_an_already_randomized_classifier():

    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork.randomized(2, 2, bounds)

    assert classifier.cardinality == 2
    assert classifier.dimension == 2
    # a fresh (non-randomized) node always starts at threshold=0.0, weights=[1.0, 1.0] -
    # confirm randomize() actually ran, not just construction
    assert any(
        node.threshold != 0.0 or list(node.input_node_weights) != [1.0, 1.0] for node in classifier.hidden_layer.nodes
    )


def test_learn_reduces_to_single_node_update_for_cardinality_one():

    dimension = 2
    bounds = square_bounds(10.0)
    learning_rate = 0.25

    for category in (0, 1):
        network = LinearClassifierNetwork.randomized(1, dimension, bounds)
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
