from random import shuffle, uniform
from typing import Any, Callable

from perceptron.model.linear_classifier_network import LinearClassifierNetwork


def random_alternating_training_data(
    size: int, classifier: LinearClassifierNetwork, max_attempts: int = 100_000
) -> list[tuple[tuple[float, ...], float]]:

    states: dict[float, list[Any]] = {0.0: [], 1.0: []}

    k: int = size // 2

    attempts = 0
    while len(states[0]) < k or len(states[1]) < k:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"failed to sample {k} examples of each class within {max_attempts} attempts "
                f"(got {len(states[0])} of class 0, {len(states[1])} of class 1) - "
                "the classifier's decision boundary likely doesn't cross its input bounds, "
                "making one class unreachable"
            )
        attempts += 1

        input = tuple(uniform(*bounds) for bounds in classifier.input_bounds)
        state: float = classifier.classify_state(input)

        if len(states[state]) < k:
            states[state].append((input, state))

    mixed = states[0] + states[1]
    shuffle(mixed)
    return mixed


def reachable_reference_and_training_data(
    cardinality: int,
    dimension: int,
    bounds: list[tuple[float, float]],
    training_set_size: int,
    regeneration_attempts: int = 20,
    max_attempts: int = 20_000,
    is_valid: Callable[[LinearClassifierNetwork], bool] | None = None,
) -> tuple[LinearClassifierNetwork, list[tuple[tuple[float, ...], float]]]:

    # higher cardinality shrinks the reference's positive region (intersection of more
    # half-planes), so some random reference classifiers make one class unreachable within
    # these bounds - regenerate the reference rather than failing on one unlucky draw.
    # is_valid, when given, is checked before the (more expensive) reachability sampling
    # below, so a caller can reject a candidate on cheap criteria (e.g. "region must be
    # bounded") without paying for training-data generation on a rejected candidate
    for _ in range(regeneration_attempts):
        reference = LinearClassifierNetwork(cardinality, dimension, bounds)
        reference.randomize()
        if is_valid is not None and not is_valid(reference):
            continue
        try:
            return reference, random_alternating_training_data(training_set_size, reference, max_attempts=max_attempts)
        except RuntimeError:
            continue

    raise RuntimeError(f"no workable cardinality={cardinality} reference classifier found within these bounds")


def smoothed_series(values: list[float], window: int = 31) -> list[float]:

    # trailing moving average - only ever looks backward, so it stays a fair comparison
    # against the raw series at every point (no look-ahead)
    smoothed = []
    for i in range(len(values)):
        segment = values[max(0, i - window + 1) : i + 1]
        smoothed.append(sum(segment) / len(segment))
    return smoothed


def class_balanced_disagreement_rate(
    reference: LinearClassifierNetwork,
    student: LinearClassifierNetwork,
    per_class_sample_count: int = 10,
    max_attempts: int = 20_000,
) -> float:

    # sampling uniformly over the bounding box would weight disagreement by each class's
    # share of the box's area, which shrinks sharply for the positive class as cardinality
    # grows - sample an equal number of each class instead, so convergence means the same
    # thing regardless of cardinality
    counts = {0.0: 0, 1.0: 0}
    disagreements = {0.0: 0, 1.0: 0}

    attempts = 0
    while counts[0.0] < per_class_sample_count or counts[1.0] < per_class_sample_count:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"failed to sample {per_class_sample_count} examples of each class within "
                f"{max_attempts} attempts - the reference classifier's decision boundary "
                "likely doesn't cross its input bounds, making one class unreachable"
            )
        attempts += 1

        state = tuple(uniform(*bounds) for bounds in reference.input_bounds)
        reference_category = reference.classify_state(state)
        if counts[reference_category] < per_class_sample_count:
            counts[reference_category] += 1
            if student.classify_state(state) != reference_category:
                disagreements[reference_category] += 1

    return (disagreements[0.0] + disagreements[1.0]) / (counts[0.0] + counts[1.0])


def train_linear_classifier_network(
    student: LinearClassifierNetwork,
    training_data: list[tuple[tuple[float, ...], float]],
    learning_rate: float = 0.25,
    epochs: int = 1,
    reference_classifier: LinearClassifierNetwork | None = None,
) -> list[tuple[int, float]]:

    iterations: int = 0
    convergence: list[tuple[int, float]] = []

    if reference_classifier:
        convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student)))

    for _ in range(epochs):
        for datum in training_data:
            (reference_state, reference_category) = datum
            student.learn(learning_rate, reference_state, reference_category)
            iterations += 1

            if reference_classifier:
                convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student)))

    return convergence
