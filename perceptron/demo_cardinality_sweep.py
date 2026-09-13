from random import uniform

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

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


def main() -> None:

    cardinalities = [1, 2, 3, 4]
    colors = ["yellow", "cyan", "magenta", "orange"]

    dimension: int = 2
    l: float = 10.0
    bounds = [(-l, l), (-l, l)]

    learning_rate: float = 0.25
    training_set_size: int = 600
    epoch_count: int = 5

    figure = new_figure("convergence by cardinality")
    axes = new_axes(figure, [(0.0, float(training_set_size * epoch_count)), (0.0, 1.0)], scaled=False)

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
        axes.plot(n, disagreement, color=color, label=f"cardinality={cardinality}")

        new_state = tuple(uniform(*b) for b in bounds)
        reference_category = reference.classify_state(new_state)
        student_category = student.classify_state(new_state)
        agreement = "agree" if reference_category == student_category else "disagree"
        print(
            f"cardinality={cardinality}: disagreement {convergence_series[0][1]:.3f} -> "
            f"{convergence_series[-1][1]:.3f}, prediction on new point: reference={reference_category}, "
            f"student={student_category} ({agreement})"
        )

    legend = axes.legend()
    legend.get_frame().set_facecolor("black")
    for text in legend.get_texts():
        text.set_color("white")

    pyplot.show()


if __name__ == "__main__":
    main()
