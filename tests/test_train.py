from perceptron.geometry import is_positive_region_bounded, square_bounds
from perceptron.train import reachable_reference_and_training_data


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
