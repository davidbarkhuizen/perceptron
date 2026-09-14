import random


def build_balanced_binary_dataset(
    dataset: list[tuple[tuple[float, ...], int]],
    target_label: int,
    class_count: int,
    rng: random.Random,
) -> list[tuple[tuple[float, ...], float]]:
    """
    Builds the "is this class target_label?" binary training set for one sub-network of an
    EnsembleBackpropClassifierNetwork: every example labeled target_label (recoded 1.0), plus a
    genuinely stratified sample of the other classes - as close to
    len(positives) // (class_count - 1) examples from *each* other class as that class has
    available (recoded 0.0), not a pooled random sample over all non-target examples (which
    would silently over/under-represent classes whose real counts differ from each other).
    Shuffled before returning.
    """

    assert class_count >= 2, f"class_count must be at least 2; got {class_count}"
    assert 0 <= target_label < class_count, f"target_label must be in [0, {class_count}); got {target_label}"

    positives = [(state, 1.0) for state, label in dataset if label == target_label]

    by_other_label: dict[int, list[tuple[float, ...]]] = {
        label: [] for label in range(class_count) if label != target_label
    }
    for state, label in dataset:
        if label != target_label:
            by_other_label[label].append(state)

    other_labels = sorted(by_other_label)
    target_negative_count = len(positives)
    base_count, remainder = divmod(target_negative_count, len(other_labels))

    negatives: list[tuple[tuple[float, ...], float]] = []
    for index, label in enumerate(other_labels):
        # spread the remainder (from integer division) across the first few classes, so the
        # total negative count matches target_negative_count as closely as availability allows
        desired = base_count + (1 if index < remainder else 0)
        available = by_other_label[label]
        sampled = rng.sample(available, min(desired, len(available)))
        negatives.extend((state, 0.0) for state in sampled)

    combined = positives + negatives
    rng.shuffle(combined)
    return combined
