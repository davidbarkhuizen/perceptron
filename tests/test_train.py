import random

import pytest

from perceptron.geometry import is_positive_region_bounded, square_bounds
from perceptron.model.linear_classifier_network import LinearClassifierNetwork
from perceptron.train import (
    random_alternating_training_data,
    reachable_reference_and_training_data,
    train_linear_classifier_network,
)


def _training_accuracy(student, training_data):
    return sum(1 for state, category in training_data if student.classify_state(state) == category) / len(
        training_data
    )


def test_generation_of_random_test_data_from_reference_classifier():

    classifier_cardinality = 1
    dimension: int = 2
    l: float = 7.0
    training_set_size: int = 50

    input_bounds = square_bounds(l, dimension)

    _, training_data = reachable_reference_and_training_data(
        classifier_cardinality, dimension, input_bounds, training_set_size
    )

    assert len(training_data) == training_set_size


def test_generation_of_random_test_data_for_non_2d_classifier():

    dimension: int = 3
    training_set_size: int = 50
    input_bounds = [(-7.0, 7.0)] * dimension

    _, training_data = reachable_reference_and_training_data(1, dimension, input_bounds, training_set_size)

    assert len(training_data) == training_set_size
    assert all(len(state) == dimension for state, _ in training_data)


def test_reachable_reference_and_training_data_returns_class_balanced_data():

    cardinality = 4
    dimension = 2
    bounds = square_bounds(10.0)
    training_set_size = 200

    reference, training_data = reachable_reference_and_training_data(cardinality, dimension, bounds, training_set_size)

    assert reference.cardinality == cardinality
    assert len(training_data) == training_set_size
    assert all(len(state) == dimension for state, _ in training_data)


def test_reachable_reference_and_training_data_respects_is_valid():

    cardinality = 4
    dimension = 2
    bounds = square_bounds(10.0)

    reference, _ = reachable_reference_and_training_data(
        cardinality, dimension, bounds, 200, regeneration_attempts=200, is_valid=is_positive_region_bounded
    )

    assert is_positive_region_bounded(reference) is True


def test_random_alternating_training_data_raises_for_an_unreachable_class():

    bounds = square_bounds(10.0)
    unreachable = LinearClassifierNetwork(1, 2, bounds)
    node = unreachable.hidden_layer.nodes[0]
    # tiny weights + a large threshold mean the decision boundary never crosses these
    # bounds, so one class can never be sampled
    node.update_input_weights([0.01, 0.01])
    node.threshold = -5.0

    with pytest.raises(RuntimeError):
        random_alternating_training_data(200, unreachable, max_attempts=200)


def test_reachable_reference_and_training_data_raises_when_no_reference_is_ever_valid():

    with pytest.raises(RuntimeError):
        reachable_reference_and_training_data(
            1, 2, square_bounds(10.0), 50, regeneration_attempts=5, is_valid=lambda classifier: False
        )


def test_train_linear_classifier_network_keeps_the_best_epoch_not_the_last():

    # this exact setup (seed, target, cardinality, required_active, epoch count) doesn't
    # converge - per-epoch training accuracy measured directly (without pocket tracking):
    # 0.736, 0.714, 0.818, 0.796, 0.818, 0.827, 0.810, 0.786, 0.773, 0.801 for epochs 0-9.
    # The raw last epoch (0.801) is worse than the best one seen (epoch 5, 0.827) - confirms
    # the student is left at the best epoch's accuracy, not whatever the last one landed on.
    random.seed(0)

    class XORTarget:
        # not linearly separable, and not representable by an AND/OR/k-of-n gate over
        # cardinality=3 half-planes either - see demo_nonrepresentable_target.py
        def __init__(self, bounds):
            self.input_bounds = bounds

        def classify_state(self, state):
            x, y = state
            return 1.0 if (x > 0) != (y > 0) else 0.0

    bounds = square_bounds(10.0)
    training_data = random_alternating_training_data(1000, XORTarget(bounds))

    student = LinearClassifierNetwork.randomized(3, 2, bounds, required_active=2)
    train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=10)

    assert _training_accuracy(student, training_data) == pytest.approx(0.827)


def test_train_linear_classifier_network_pocket_tracking_is_a_no_op_when_it_converges():

    # when training does converge, the best epoch and the last epoch coincide, so pocket
    # tracking shouldn't change the well-established convergence behavior at all
    random.seed(0)

    bounds = square_bounds(10.0)
    reference, training_data = reachable_reference_and_training_data(1, 2, bounds, 400)
    student = LinearClassifierNetwork.randomized(1, 2, bounds)

    train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=5)

    assert _training_accuracy(student, training_data) >= 0.99
