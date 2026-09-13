from matplotlib import lines, pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from perceptron.geometry import reference_positive_region_polygon
from perceptron.model.linear_classifier_network import LinearClassifierNetwork


def plot_linear_classifier_network(
    axes: Axes,
    classifier: LinearClassifierNetwork,
    plotting_resolution: int = 100,
    color: str = "purple",
    x_bounds: tuple[float, float] | None = None,
):
    assert classifier.dimension == 2, (
        "plot_linear_classifier_network only supports 2D classifiers - each hidden node's "
        f"decision line is drawn in the x/y plane; got dimension={classifier.dimension}"
    )

    x_min, x_max = x_bounds if x_bounds else classifier.input_bounds[0]

    x_interval_size = x_max - x_min
    x_step_size = x_interval_size / float(plotting_resolution)
    x_ = [x_min + (i * x_step_size) for i in range(plotting_resolution)]

    for node in classifier.hidden_layer.nodes:
        a = node.input_node_weights[0]
        b = node.input_node_weights[1]
        c = node.threshold

        if b == 0.0 and a == 0.0:
            # a*x + b*y + c = 0 doesn't depend on position at all (the node is either always
            # active or always inactive, everywhere) - there's no line to draw for it
            continue
        elif b == 0.0:
            # a*x + c = 0 is vertical (no y term to solve for) - draw x = -c/a spanning the
            # axes' current y-range instead of dividing by the zero b below
            x_line = -c / a
            y_min, y_max = axes.get_ylim()
            line_graph = lines.Line2D([x_line, x_line], [y_min, y_max], color=color)
        else:
            y_ = [-1.0 * (a * x + c) / b for x in x_]
            line_graph = lines.Line2D(x_, y_, color=color)

        axes.add_line(line_graph)


def reference_region_bounds(
    classifier: LinearClassifierNetwork,
    fallback_bounds: list[tuple[float, float]],
    margin_fraction: float = 0.1,
) -> list[tuple[float, float]]:
    """
    Returns fallback_bounds expanded just enough to fully contain the classifier's positive
    region when that region is bounded (see geometry.reference_positive_region_polygon);
    returns fallback_bounds unchanged when it's unbounded (or empty).
    """

    polygon = reference_positive_region_polygon(classifier)
    if not polygon:
        return fallback_bounds

    region_x = [x for x, _ in polygon]
    region_y = [y for _, y in polygon]
    region_x_min, region_x_max = min(region_x), max(region_x)
    region_y_min, region_y_max = min(region_y), max(region_y)

    margin_x = margin_fraction * max(region_x_max - region_x_min, 1.0e-9)
    margin_y = margin_fraction * max(region_y_max - region_y_min, 1.0e-9)

    (fallback_x_min, fallback_x_max), (fallback_y_min, fallback_y_max) = fallback_bounds

    return [
        (min(fallback_x_min, region_x_min - margin_x), max(fallback_x_max, region_x_max + margin_x)),
        (min(fallback_y_min, region_y_min - margin_y), max(fallback_y_max, region_y_max + margin_y)),
    ]


def plot_training_data(axes: Axes, training_data: list[tuple[tuple[float, float], float]]):

    markers = [".", "x"]
    colors = ["blue", "yellow"]

    categories: list[tuple[int, list[tuple[float, float]]]] = []

    # sorted rather than a bare set(): iteration order over a set isn't a guaranteed
    # contract, so relying on it would make which class gets which marker/color
    # implementation-defined rather than a predictable 0.0 -> first, 1.0 -> second
    for category_value in sorted(set(output_value for (_, output_value) in training_data)):
        categories.append(
            (category_value, [xy for (xy, output_value) in training_data if output_value == category_value])
        )

    for i in range(len(categories)):
        (category_value, values) = categories[i]
        marker = markers[i]
        color = colors[i]
        x, y = zip(*[(x_[0], x_[1]) for x_ in values])
        axes.plot(x, y, marker, color=color)


def disagreement_axis_bounds(x_max: float, log: bool = False) -> list[tuple[float, float]]:
    # a disagreement rate is always in [0, 1]; on a log-scaled axis 0.0 has no position, so
    # floor it just above zero instead
    return [(0.0, x_max), (1.0e-3, 1.0) if log else (0.0, 1.0)]


def new_figure(label: str) -> Figure:
    figure = pyplot.figure(label)
    figure.patch.set_facecolor("xkcd:black")
    return figure


def new_axes(figure: Figure, bounds: list[tuple[float, float]] | None = None, scaled=True) -> Axes:

    axes = figure.add_subplot(111)
    axes.set_facecolor("xkcd:black")

    axes.grid(True, which="both")

    if bounds:
        axes.set_xlim(bounds[0])
        axes.set_ylim(bounds[1])

    if scaled:
        axes.set_aspect("equal", adjustable="box")

    axes.spines["bottom"].set_color("white")
    axes.spines["top"].set_color("white")
    axes.spines["left"].set_color("white")
    axes.spines["right"].set_color("white")

    axes.xaxis.label.set_color("white")
    axes.yaxis.label.set_color("white")
    axes.tick_params(axis="x", colors="white")
    axes.tick_params(axis="y", colors="white")

    return axes
