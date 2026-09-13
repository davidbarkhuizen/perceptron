from random import shuffle, uniform
from typing import Any

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


def classification_disagreement_rate(
    reference: LinearClassifierNetwork,
    student: LinearClassifierNetwork,
    sample_count: int = 30,
) -> float:

    disagreements = 0
    for _ in range(sample_count):
        state = tuple(uniform(*bounds) for bounds in reference.input_bounds)
        if reference.classify_state(state) != student.classify_state(state):
            disagreements += 1

    return disagreements / sample_count


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
        convergence.append((iterations, classification_disagreement_rate(reference_classifier, student)))

    for _ in range(epochs):
        for datum in training_data:
            (reference_state, reference_category) = datum
            student.learn(learning_rate, reference_state, reference_category)
            iterations += 1

            if reference_classifier:
                convergence.append((iterations, classification_disagreement_rate(reference_classifier, student)))

    return convergence
