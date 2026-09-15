import random

import numpy as np
import pytest

from perceptron.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from perceptron.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)

DIMENSION = 6
LAYER_SIZES = [5]
CLASS_COUNT = 3


def _matching_networks(rng: random.Random):
    # numpy's RNG and Python's random module are separate streams, so two independently
    # randomize()d networks are never meaningfully comparable - initial weights are always
    # forced identical explicitly here instead (see docs/vectorized-array-classes.md's own
    # "RNG-state caveat" note), the array-vs-node analogue of tests/test_learn_batch.py's own
    # restore-then-compare-snapshots idiom.
    node_network = MultiClassBackpropClassifierNetwork(
        LAYER_SIZES, DIMENSION, [(-10.0, 10.0)] * DIMENSION, CLASS_COUNT
    )
    array_network = VectorizedMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT)

    previous_size = DIMENSION
    for layer_index, size in enumerate([*LAYER_SIZES, CLASS_COUNT]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = np.array(weights)
        array_network.layers[layer_index].b = np.array(biases)

        previous_size = size

    return node_network, array_network


def _assert_networks_match(node_network, array_network, rtol=1e-9, atol=1e-9):
    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        expected_W = np.array([node.input_node_weights for node in node_layer.nodes])
        expected_b = np.array([node.bias for node in node_layer.nodes])
        assert np.allclose(array_layer.W, expected_W, rtol=rtol, atol=atol)
        assert np.allclose(array_layer.b, expected_b, rtol=rtol, atol=atol)


def test_predict_probabilities_matches_across_a_random_sweep():

    rng = random.Random(0)
    node_network, array_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        expected = node_network.predict_probabilities(state)
        actual = array_network.predict_probabilities(state)
        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_classify_state_matches_across_a_random_sweep():

    rng = random.Random(1)
    node_network, array_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        assert array_network.classify_state(state) == node_network.classify_state(state)


def test_learn_matches_after_every_step_not_just_at_the_end():

    # one silently-wrong intermediate step should fail loudly rather than being averaged away
    # by many steps - per docs/vectorized-array-classes.md's own "required regression gate"
    rng = random.Random(2)
    node_network, array_network = _matching_networks(rng)
    learning_rate = 0.3

    for step in range(100):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        category = rng.randrange(CLASS_COUNT)

        node_network.learn(learning_rate, state, category)
        array_network.learn(learning_rate, state, category)

        _assert_networks_match(node_network, array_network)


def test_learn_batch_matches_after_every_batch_not_just_at_the_end():

    rng = random.Random(3)
    node_network, array_network = _matching_networks(rng)
    learning_rate = 0.3
    batch_size = 8

    for _ in range(20):
        batch = [
            (tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION)), rng.randrange(CLASS_COUNT))
            for _ in range(batch_size)
        ]

        node_network.learn_batch(learning_rate, batch)
        array_network.learn_batch(learning_rate, batch)

        _assert_networks_match(node_network, array_network)


def test_randomized_builds_a_usable_network():

    network = VectorizedMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    state = tuple(0.1 * i for i in range(DIMENSION))

    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == CLASS_COUNT
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert 0 <= network.classify_state(state) < CLASS_COUNT


def test_snapshot_restore_round_trips_weights():

    network = VectorizedMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    snapshot = network.snapshot()

    other = VectorizedMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert np.array_equal(W1, W2)
        assert np.array_equal(b1, b2)


def test_save_load_round_trips_weights_and_predictions(tmp_path):

    network = VectorizedMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    path = str(tmp_path / "vectorized_model.json")
    network.save(path)

    loaded = VectorizedMultiClassBackpropClassifierNetwork.load(path)

    assert loaded.layer_sizes == network.layer_sizes
    assert loaded.dimension == network.dimension
    assert loaded.class_count == network.class_count

    state = tuple(0.1 * i for i in range(DIMENSION))
    assert loaded.predict_probabilities(state) == pytest.approx(network.predict_probabilities(state))


def test_construction_rejects_invalid_arguments():

    with pytest.raises(AssertionError):
        VectorizedMultiClassBackpropClassifierNetwork([], DIMENSION, CLASS_COUNT)

    with pytest.raises(AssertionError):
        VectorizedMultiClassBackpropClassifierNetwork([0], DIMENSION, CLASS_COUNT)

    with pytest.raises(AssertionError):
        VectorizedMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, class_count=1)


def test_learn_batch_rejects_an_empty_batch():

    network = VectorizedMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    with pytest.raises(AssertionError):
        network.learn_batch(0.1, [])
