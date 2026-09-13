from random import uniform

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot
from matplotlib.axes import Axes

from perceptron.graphics.chart import new_axes, new_figure
from perceptron.model.linear_classifier_network import LinearClassifierNetwork
from perceptron.train import random_alternating_training_data, train_linear_classifier_network


def _reference_and_training_data(
    cardinality: int,
    dimension: int,
    bounds: list[tuple[float, float]],
    training_set_size: int,
    regeneration_attempts: int = 20,
) -> tuple[LinearClassifierNetwork, list[tuple[tuple[float, ...], float]]]:

    # higher cardinality shrinks the reference's positive region (intersection of more
    # half-planes), so some random reference classifiers make one class unreachable within
    # these bounds - regenerate the reference rather than failing the whole sweep on one
    # unlucky draw
    for _ in range(regeneration_attempts):
        reference = LinearClassifierNetwork(cardinality, dimension, bounds)
        reference.randomize()
        try:
            return reference, random_alternating_training_data(training_set_size, reference, max_attempts=20_000)
        except RuntimeError:
            continue

    raise RuntimeError(f"no workable cardinality={cardinality} reference classifier found within these bounds")


def _smoothed(values: list[float], window: int = 31) -> list[float]:

    # trailing moving average - only ever looks backward, so it stays a fair comparison
    # against the raw series at every point (no look-ahead)
    smoothed = []
    for i in range(len(values)):
        segment = values[max(0, i - window + 1) : i + 1]
        smoothed.append(sum(segment) / len(segment))
    return smoothed


def _plot_convergence(axes: Axes, results: list[tuple[int, str, list[int], list[float]]]) -> None:

    for cardinality, color, n, disagreement in results:
        axes.plot(n, disagreement, color=color, label=f"cardinality={cardinality}")

    legend = axes.legend()
    legend.get_frame().set_facecolor("black")
    for text in legend.get_texts():
        text.set_color("white")


def main() -> None:

    cardinalities = [1, 2, 3, 4]
    colors = ["yellow", "cyan", "magenta", "orange"]

    dimension: int = 2
    l: float = 10.0
    bounds = [(-l, l), (-l, l)]

    learning_rate: float = 0.25
    training_set_size: int = 600
    epoch_count: int = 5

    results: list[tuple[int, str, list[int], list[float]]] = []

    for cardinality, color in zip(cardinalities, colors):
        reference, training_data = _reference_and_training_data(cardinality, dimension, bounds, training_set_size)

        student = LinearClassifierNetwork(cardinality, dimension, bounds)
        student.randomize()

        convergence_series = train_linear_classifier_network(
            student,
            training_data,
            learning_rate=learning_rate,
            epochs=epoch_count,
            reference_classifier=reference,
        )

        n = [x[0] for x in convergence_series]
        disagreement = [x[1] for x in convergence_series]
        results.append((cardinality, color, n, disagreement))

        new_state = tuple(uniform(*b) for b in bounds)
        reference_category = reference.classify_state(new_state)
        student_category = student.classify_state(new_state)
        agreement = "agree" if reference_category == student_category else "disagree"
        print(
            f"cardinality={cardinality}: disagreement {convergence_series[0][1]:.3f} -> "
            f"{convergence_series[-1][1]:.3f}, prediction on new point: reference={reference_category}, "
            f"student={student_category} ({agreement})"
        )

    x_max = float(training_set_size * epoch_count)

    print(
        "linear-scale convergence chart: x-axis = training iteration (pooled across epochs), "
        "y-axis = disagreement rate between each cardinality's reference and student "
        "classifier - one curve per cardinality, lower is better"
    )

    linear_figure = new_figure("convergence by cardinality")
    linear_axes = new_axes(linear_figure, [(0.0, x_max), (0.0, 1.0)], scaled=False)
    _plot_convergence(linear_axes, results)

    print(
        "log-scale convergence chart: the same per-cardinality disagreement-rate curves as "
        "above, with a log-scaled y-axis - useful for comparing how fast each cardinality "
        "converges, since disagreement tends to drop roughly exponentially; a curve stops "
        "early once its disagreement reaches exactly zero, which has no position on a log axis"
    )

    log_figure = new_figure("convergence by cardinality (log scale)")
    log_axes = new_axes(log_figure, [(0.0, x_max), (1e-3, 1.0)], scaled=False)
    log_axes.set_yscale("log")
    _plot_convergence(log_axes, results)

    smoothed_results = [
        (cardinality, color, n, _smoothed(disagreement)) for cardinality, color, n, disagreement in results
    ]

    print(
        "smoothed convergence chart: the same per-cardinality disagreement-rate curves, each "
        "passed through a trailing moving average - makes the underlying trend easier to see "
        "through the sampling noise from classification_disagreement_rate's small per-checkpoint "
        "sample size"
    )

    smoothed_figure = new_figure("convergence by cardinality (smoothed)")
    smoothed_axes = new_axes(smoothed_figure, [(0.0, x_max), (0.0, 1.0)], scaled=False)
    _plot_convergence(smoothed_axes, smoothed_results)

    pyplot.show()


if __name__ == "__main__":
    main()
