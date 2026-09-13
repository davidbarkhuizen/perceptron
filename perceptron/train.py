from random import shuffle, uniform
from typing import Any

from perceptron.model.linear_classifier_network import LinearClassifierNetwork


def random_alternating_training_data(
    size: int, classifier: LinearClassifierNetwork, max_attempts: int = 100_000
) -> list[tuple[tuple[float, float], int]]:

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

        input = (uniform(*classifier.input_bounds[0]), uniform(*classifier.input_bounds[1]))
        state: float = classifier.classify_state(input)

        if len(states[state]) < k:
            states[state].append((input, state))

    mixed = states[0] + states[1]
    shuffle(mixed)
    return mixed


def train_linear_classifier_network(
    student: LinearClassifierNetwork,
    training_data: list[tuple[tuple[float, float], int]],
    learning_rate: float = 0.25,
    epochs: int = 1,
    reference_classifier: LinearClassifierNetwork | None = None,
) -> list[tuple[int, float]]:

    iterations: int = 0
    convergence: list[tuple[int, float]] = []

    if reference_classifier:
        convergence.append((iterations, reference_classifier.distance(student)))

    for _ in range(epochs):
        for datum in training_data:
            (reference_state, reference_category) = datum
            student.learn(learning_rate, reference_state, reference_category)
            iterations += 1

            if reference_classifier:
                convergence.append((iterations, reference_classifier.distance(student)))

    return convergence
