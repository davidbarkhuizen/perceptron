import multiprocessing
import os
import random

from perceptron.model.backprop_classifier_network import BackpropClassifierNetwork
from perceptron.model.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork
from perceptron.train import TrainingDiagnostic, train_linear_classifier_network


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


def _train_one_classifier(
    args: tuple[int, list[tuple[tuple[float, ...], float]], list[int], int, list[tuple[float, float]], float, int, int | None],
) -> tuple[int, list[list[tuple[list[float], float]]], TrainingDiagnostic]:
    """
    The multiprocessing.Pool worker - a plain module-level function, required for picklability.
    Trains one class's binary BackpropClassifierNetwork completely independently: no state is
    shared with any other worker, which is what makes this genuinely (not just approximately)
    parallelizable - see docs/research-and-analysis.md's "parallelizing MNIST training" entry.

    Explicitly seeds this process's own random state before building anything - confirmed
    directly this session that fork-based multiprocessing workers are not guaranteed to diverge
    from each other's global random state on their own before their first random call, so
    relying on incidental post-fork divergence would risk correlated (or even identical) initial
    weights across sub-networks. seed=None still calls random.seed(None), which reseeds from the
    OS's own entropy source independently per process - safe, just not reproducible.
    """

    label, binary_dataset, layer_sizes, dimension, input_bounds, learning_rate, epochs, seed = args

    random.seed(seed)
    student = BackpropClassifierNetwork.randomized(layer_sizes, dimension, input_bounds)
    result = train_linear_classifier_network(student, binary_dataset, learning_rate=learning_rate, epochs=epochs)

    return label, student.snapshot(), result.diagnostic


def train_ensemble_parallel(
    dataset: list[tuple[tuple[float, ...], int]],
    class_count: int,
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    learning_rate: float,
    epochs: int,
    worker_count: int | None = None,
    seed: int | None = None,
) -> tuple[EnsembleBackpropClassifierNetwork, dict[int, TrainingDiagnostic]]:
    """
    Builds all class_count balanced binary datasets, then trains one BackpropClassifierNetwork
    per class completely independently - dispatched across a multiprocessing.Pool, since
    nothing needs to be synchronized between them (unlike the data-parallel weight-averaging
    approach rejected in docs/research-and-analysis.md, this has no communication cost beyond
    the one-time dispatch and final collection).

    seed, when given, makes the whole run reproducible: it seeds a single random.Random used for
    every dataset's stratified sampling (in class order, so the sequence is deterministic) and
    to derive each job's own per-worker seed - not the same rng instance as any worker's (those
    run in separate processes with their own random module state).
    """

    rng = random.Random(seed)

    jobs = []
    for label in range(class_count):
        binary_dataset = build_balanced_binary_dataset(dataset, label, class_count, rng)
        job_seed = rng.randrange(2**31) if seed is not None else None
        jobs.append((label, binary_dataset, layer_sizes, dimension, input_bounds, learning_rate, epochs, job_seed))

    with multiprocessing.Pool(worker_count or os.cpu_count()) as pool:
        results = pool.map(_train_one_classifier, jobs)

    results.sort(key=lambda result: result[0])

    classifiers = []
    diagnostics: dict[int, TrainingDiagnostic] = {}
    for label, snapshot, diagnostic in results:
        student = BackpropClassifierNetwork(layer_sizes, dimension, input_bounds)
        student.restore(snapshot)
        classifiers.append(student)
        diagnostics[label] = diagnostic

    return EnsembleBackpropClassifierNetwork(classifiers), diagnostics
