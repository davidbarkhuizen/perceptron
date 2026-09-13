from random import uniform

from perceptron.model.linear_classifier_network import LinearClassifierNetwork


def compare_on_random_point(
    reference: LinearClassifierNetwork, student: LinearClassifierNetwork
) -> tuple[tuple[float, ...], float, float]:

    state = tuple(uniform(*bounds) for bounds in reference.input_bounds)
    return state, reference.classify_state(state), student.classify_state(state)


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
