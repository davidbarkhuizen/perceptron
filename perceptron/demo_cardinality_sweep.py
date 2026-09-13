import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot
from matplotlib.axes import Axes

from perceptron.graphics.chart import disagreement_axis_bounds, new_axes, new_figure
from perceptron.model.linear_classifier_network import LinearClassifierNetwork
from perceptron.train import (
    compare_on_random_point,
    reachable_reference_and_training_data,
    smoothed_series,
    train_linear_classifier_network,
)


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
        reference, training_data = reachable_reference_and_training_data(
            cardinality, dimension, bounds, training_set_size
        )

        student = LinearClassifierNetwork.randomized(cardinality, dimension, bounds)

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

        new_state, reference_category, student_category = compare_on_random_point(reference, student)
        agreement = "agree" if reference_category == student_category else "disagree"
        print(
            f"cardinality={cardinality}: disagreement {convergence_series[0][1]:.3f} -> "
            f"{convergence_series[-1][1]:.3f}, prediction on new point: reference={reference_category}, "
            f"student={student_category} ({agreement})"
        )

    x_max = float(training_set_size * epoch_count)

    smoothed_results = [
        (cardinality, color, n, smoothed_series(disagreement)) for cardinality, color, n, disagreement in results
    ]

    print(
        "smoothed convergence chart: x-axis = training iteration (pooled across epochs), "
        "y-axis = each cardinality's disagreement-rate series passed through a trailing "
        "moving average - makes the underlying trend easier to see through the sampling "
        "noise from class_balanced_disagreement_rate's small per-class sample count"
    )

    smoothed_figure = new_figure("convergence by cardinality (smoothed)")
    smoothed_axes = new_axes(smoothed_figure, disagreement_axis_bounds(x_max), scaled=False)
    _plot_convergence(smoothed_axes, smoothed_results)

    print(
        "smoothed log-scale convergence chart: the same smoothed curves as above, with a "
        "log-scaled y-axis - useful for comparing how fast each cardinality converges, "
        "since disagreement tends to drop roughly exponentially; a curve stops early once "
        "its disagreement reaches exactly zero, which has no position on a log axis"
    )

    smoothed_log_figure = new_figure("convergence by cardinality (smoothed, log scale)")
    smoothed_log_axes = new_axes(smoothed_log_figure, disagreement_axis_bounds(x_max, log=True), scaled=False)
    smoothed_log_axes.set_yscale("log")
    _plot_convergence(smoothed_log_axes, smoothed_results)

    pyplot.show()


if __name__ == "__main__":
    main()
