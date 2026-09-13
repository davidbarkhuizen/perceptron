from random import uniform

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot
from matplotlib.axes import Axes

from perceptron.graphics.chart import new_axes, new_figure, plot_linear_classifier_network, plot_training_data
from perceptron.model.linear_classifier_network import LinearClassifierNetwork
from perceptron.train import random_alternating_training_data, smoothed_series, train_linear_classifier_network


def main() -> None:

    classifier_cardinality: int = 2
    dimension: int = 2
    l: float = 10.0

    learning_rate: float = 0.25
    training_set_size: int = 1000
    epoch_count: int = 1

    x_min, x_max = -l, l
    y_min, y_max = -l, l
    input_bounds = [(x_min, x_max), (y_min, y_max)]

    # generate a (random) reference classifier network
    #
    reference_classifier = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    reference_classifier.randomize()

    # use the reference classifier to produce a set of training data
    #
    training_data = random_alternating_training_data(training_set_size, reference_classifier)

    # generate a new random classifier network for training
    #
    student_classifier = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    student_classifier.randomize()

    convergence_series: list[tuple[int, float]] = train_linear_classifier_network(
        student_classifier,
        training_data,
        learning_rate=learning_rate,
        epochs=epoch_count,
        reference_classifier=reference_classifier,
    )

    # use the trained student to classify a fresh point, never seen during training
    #
    new_state = tuple(uniform(*bounds) for bounds in input_bounds)
    reference_category = reference_classifier.classify_state(new_state)
    student_category = student_classifier.classify_state(new_state)
    agreement = "agree" if reference_category == student_category else "disagree"
    print(f"prediction on new point {new_state}: reference={reference_category}, student={student_category} ({agreement})")

    n: list[int] = [x[0] for x in convergence_series]
    disagreement: list[float] = smoothed_series([x[1] for x in convergence_series])

    print(
        "convergence chart (linear scale): x-axis = training iteration, y-axis = "
        "disagreement rate (fraction of sampled points where the student's classification "
        "differs from the reference's), smoothed with a trailing moving average - lower "
        "means the student more closely matches the reference"
    )

    linear_convergence_figure = new_figure("convergence (linear scale)")

    linear_convergence_bounds = [
        (0.0, float(training_set_size)),
        (0.0, 1.0),
    ]

    linear_convergence_axes: Axes = new_axes(linear_convergence_figure, linear_convergence_bounds, scaled=False)
    linear_convergence_axes.plot(n, disagreement)

    pyplot.get_current_fig_manager().window.wm_geometry("+800+0")
    pyplot.show(block=False)

    print(
        "convergence chart (log scale): the same smoothed disagreement-rate series, "
        "log-scaled - useful for seeing how fast the student converges, since disagreement "
        "tends to drop roughly exponentially; the curve stops early once disagreement "
        "reaches exactly zero, which has no position on a log axis"
    )

    log_convergence_figure = new_figure("convergence (log scale)")

    log_convergence_bounds = [
        (0.0, float(training_set_size)),
        (1e-3, 1.0),
    ]

    log_convergence_axes: Axes = new_axes(log_convergence_figure, log_convergence_bounds, scaled=False)
    log_convergence_axes.set_yscale("log")
    log_convergence_axes.plot(n, disagreement)

    pyplot.get_current_fig_manager().window.wm_geometry("+800+500")
    pyplot.show(block=False)

    # ---------------------

    print(
        "decision-boundary chart: training data points colored by class, the reference "
        "classifier's hyperplane(s) in green, and the trained student's in purple - the "
        "closer the purple lines are to the green ones, the more closely the student has "
        "learned the reference's decision boundary"
    )

    training_data_figure = new_figure("perceptrons (reference, student) with training data")
    axes = new_axes(training_data_figure, student_classifier.input_bounds)

    axes.set_xlim(input_bounds[0])
    axes.set_ylim(input_bounds[1])

    plot_training_data(axes, training_data)
    plot_linear_classifier_network(axes, reference_classifier, color="green")
    plot_linear_classifier_network(axes, student_classifier, color="purple")

    pyplot.show()


if __name__ == "__main__":
    main()
