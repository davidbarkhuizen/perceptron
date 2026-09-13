from random import shuffle, uniform
from typing import Any, Callable

from perceptron.evaluate import class_balanced_disagreement_rate
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
        reference = LinearClassifierNetwork.randomized(cardinality, dimension, bounds)
        if is_valid is not None and not is_valid(reference):
            continue
        try:
            return reference, random_alternating_training_data(training_set_size, reference, max_attempts=max_attempts)
        except RuntimeError:
            continue

    raise RuntimeError(f"no workable cardinality={cardinality} reference classifier found within these bounds")


def _training_accuracy(
    student: LinearClassifierNetwork, training_data: list[tuple[tuple[float, ...], float]]
) -> float:
    correct = sum(1 for state, category in training_data if student.classify_state(state) == category)
    return correct / len(training_data)


class TrainingDiagnostic:
    """
    Summarizes how a train_linear_classifier_network() run's training-data accuracy
    trajectory behaved, since there's no guarantee it converges (see docs/structure.md) and
    eyeballing a chart is otherwise the only way to tell converged from plateaued from still
    improving.
    """

    def __init__(
        self, epoch_training_accuracies: list[float], best_epoch_index: int, best_training_accuracy: float
    ) -> None:
        self.epoch_training_accuracies = epoch_training_accuracies
        # best_epoch_index is -1 (rather than an index into epoch_training_accuracies) when
        # no epoch ever beat the untrained starting point's own accuracy
        self.best_epoch_index = best_epoch_index
        self.best_training_accuracy = best_training_accuracy

    @property
    def converged(self) -> bool:
        # every training example correctly classified
        return self.best_training_accuracy >= 1.0

    @property
    def plateaued(self) -> bool:
        # the best epoch wasn't the last one - later epochs never improved on it
        return not self.converged and self.best_epoch_index < len(self.epoch_training_accuracies) - 1

    @property
    def still_improving(self) -> bool:
        # the last epoch was still the best one seen, but training hasn't converged yet -
        # more epochs might help
        return not self.converged and not self.plateaued


class ConvergenceSeries(list):
    """
    Exactly the list of (iteration, disagreement_rate) pairs train_linear_classifier_network
    has always returned - every existing use (indexing, iterating, len(), list
    comprehensions) keeps working unchanged - plus a `.diagnostic` (a TrainingDiagnostic) for
    callers that want to know whether training converged, plateaued, or was still improving.
    """

    diagnostic: TrainingDiagnostic


def train_linear_classifier_network(
    student: LinearClassifierNetwork,
    training_data: list[tuple[tuple[float, ...], float]],
    learning_rate: float = 0.25,
    epochs: int = 1,
    reference_classifier: LinearClassifierNetwork | None = None,
) -> ConvergenceSeries:
    """
    Trains student in place over training_data for the given number of epochs.

    There's no guarantee this converges (see docs/structure.md) - training accuracy can
    oscillate rather than settle, especially once the target isn't exactly representable at
    student's cardinality/required_active. So rather than leaving student wherever the last
    epoch happened to land, it's left at whichever epoch's end had the best training-data
    accuracy seen (a pocket-algorithm-style "keep the best, not the latest" snapshot) - a
    strict improvement when training does converge (the best epoch is then the last one, so
    this is a no-op), and a real difference when it doesn't. See the returned
    ConvergenceSeries's .diagnostic for whether this run converged, plateaued, or was still
    improving.

    The returned series itself holds the disagreement-rate series against
    reference_classifier, sampled before training and after every single learning step, when
    reference_classifier is given (used for plotting a convergence curve) - this reflects the
    raw, unrolled-back trajectory actually taken during training, not the final pocketed
    student.
    """

    iterations: int = 0
    convergence: list[tuple[int, float]] = []

    if reference_classifier:
        convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student)))

    best_snapshot = student.hidden_layer_snapshot()
    best_training_accuracy = _training_accuracy(student, training_data)
    best_epoch_index = -1  # -1: the untrained starting point was never beaten
    epoch_training_accuracies: list[float] = []

    for epoch_index in range(epochs):
        for datum in training_data:
            (reference_state, reference_category) = datum
            student.learn(learning_rate, reference_state, reference_category)
            iterations += 1

            if reference_classifier:
                convergence.append((iterations, class_balanced_disagreement_rate(reference_classifier, student)))

        training_accuracy = _training_accuracy(student, training_data)
        epoch_training_accuracies.append(training_accuracy)
        if training_accuracy > best_training_accuracy:
            best_training_accuracy = training_accuracy
            best_epoch_index = epoch_index
            best_snapshot = student.hidden_layer_snapshot()

    student.restore_hidden_layer(best_snapshot)

    result = ConvergenceSeries(convergence)
    result.diagnostic = TrainingDiagnostic(epoch_training_accuracies, best_epoch_index, best_training_accuracy)
    return result
