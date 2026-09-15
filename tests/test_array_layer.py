import random

import numpy as np
import pytest

from perceptron.model.array_layer import ArrayLayer, sigmoid
from perceptron.model.backprop_layer import BackpropLayer
from perceptron.model.backprop_node import sigmoid as node_sigmoid
from perceptron.model.state_layer import StateLayer


def test_sigmoid_matches_node_sigmoid_across_a_random_sweep_including_the_overflow_boundary():

    # backprop_node.sigmoid's own docstring pins math.exp(710) as raising OverflowError and
    # math.exp(700) as not - both sides of that boundary, plus the ordinary range, get checked
    # here so numpy's inf-based overflow path is confirmed to land on the same limiting value,
    # not assumed from the formulas looking equivalent (see array_layer.sigmoid's docstring).
    rng = random.Random(0)
    z_values = [rng.uniform(-50.0, 50.0) for _ in range(200)]
    z_values += [-700.0, -709.0, -710.0, -1000.0, -1e10, 700.0, 709.0, 710.0, 1000.0, 0.0]

    for z in z_values:
        expected = node_sigmoid(z)
        actual = float(sigmoid(np.array([z]))[0])
        assert actual == pytest.approx(expected, abs=1e-12)


def _snapshot_to_array_layer(backprop_layer: BackpropLayer) -> ArrayLayer:
    array_layer = ArrayLayer(backprop_layer.size, len(backprop_layer.input_layer.nodes))
    snapshot = backprop_layer.snapshot_state()
    array_layer.W = np.array([weights for weights, _bias in snapshot])
    array_layer.b = np.array([bias for _weights, bias in snapshot])
    return array_layer


def test_forward_matches_backprop_layer_across_a_random_sweep():

    rng = random.Random(1)
    dimension = 5
    size = 4

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        backprop_layer = BackpropLayer(size, state_layer)

        weights = [[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)]
        biases = [rng.uniform(-3.0, 3.0) for _ in range(size)]
        for node, node_weights, bias in zip(backprop_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_layer = _snapshot_to_array_layer(backprop_layer)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        backprop_layer.forward()
        expected = [node.value() for node in backprop_layer.nodes]

        actual = array_layer.forward(np.array(x))

        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_forward_batch_matches_forward_run_once_per_row_and_stacked():

    rng = random.Random(2)
    dimension = 6
    size = 3
    batch_size = 8

    array_layer = ArrayLayer(size, dimension)
    array_layer.W = np.array([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    array_layer.b = np.array([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = np.array([[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)])

    expected = np.stack([array_layer.forward(x) for x in X])

    actual = array_layer.forward_batch(X)

    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
