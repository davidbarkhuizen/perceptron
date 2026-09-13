from matplotlib import lines, pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from perceptron.model.linear_classifier_network import LinearClassifierNetwork


def plot_linear_classifier_network(
    axes: Axes,
    classifier: LinearClassifierNetwork,
    plotting_resolution: int = 100,
    color: str = "purple",
    x_bounds: tuple[float, float] | None = None,
):

    x_min, x_max = x_bounds if x_bounds else classifier.input_bounds[0]

    x_interval_size = x_max - x_min
    x_step_size = x_interval_size / float(plotting_resolution)
    x_ = [x_min + (i * x_step_size) for i in range(plotting_resolution)]

    for node in classifier.hidden_layer.nodes:
        a = node.input_node_weights[0]
        b = node.input_node_weights[1]
        c = node.threshold

        y_ = [-1.0 * (a * x + c) / b for x in x_]

        line_graph = lines.Line2D(x_, y_, color=color)

        axes.add_line(line_graph)


def _clip_polygon_by_halfplane(
    polygon: list[tuple[float, float]], a: float, b: float, c: float
) -> list[tuple[float, float]]:

    # Sutherland-Hodgman clipping against the half-plane a*x + b*y + c > 0 (the same
    # inequality AssociationNode.z() > 0 tests for a hidden node's weights/threshold)
    def signed_distance(point: tuple[float, float]) -> float:
        return a * point[0] + b * point[1] + c

    output: list[tuple[float, float]] = []

    for i in range(len(polygon)):
        current = polygon[i]
        previous = polygon[i - 1]

        current_distance = signed_distance(current)
        previous_distance = signed_distance(previous)

        current_inside = current_distance > 0.0
        previous_inside = previous_distance > 0.0

        if current_inside != previous_inside:
            t = previous_distance / (previous_distance - current_distance)
            output.append(
                (
                    previous[0] + t * (current[0] - previous[0]),
                    previous[1] + t * (current[1] - previous[1]),
                )
            )

        if current_inside:
            output.append(current)

    return output


def _reference_positive_region_polygon(
    classifier: LinearClassifierNetwork, huge: float = 1.0e6
) -> list[tuple[float, float]]:
    """
    The classifier's positive region is the intersection of its hidden nodes' half-planes,
    which may be bounded (closed, e.g. a triangle or other convex polygon) or unbounded
    (e.g. any single half-plane, or several whose intersection still extends to infinity).

    Returns the region's polygon vertices when it's bounded; an empty list when it's
    unbounded or empty.
    """

    polygon: list[tuple[float, float]] = [(-huge, -huge), (huge, -huge), (huge, huge), (-huge, huge)]

    for node in classifier.hidden_layer.nodes:
        a, b, c = node.input_node_weights[0], node.input_node_weights[1], node.threshold
        polygon = _clip_polygon_by_halfplane(polygon, a, b, c)
        if not polygon:
            return []

    if any(abs(x) >= huge * 0.99 or abs(y) >= huge * 0.99 for x, y in polygon):
        return []

    return polygon


def is_positive_region_bounded(classifier: LinearClassifierNetwork) -> bool:
    return bool(_reference_positive_region_polygon(classifier))


def reference_region_bounds(
    classifier: LinearClassifierNetwork,
    fallback_bounds: list[tuple[float, float]],
    margin_fraction: float = 0.1,
) -> list[tuple[float, float]]:
    """
    Returns fallback_bounds expanded just enough to fully contain the classifier's positive
    region when that region is bounded (see _reference_positive_region_polygon); returns
    fallback_bounds unchanged when it's unbounded (or empty).
    """

    polygon = _reference_positive_region_polygon(classifier)
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

    for category_value in set([output_value for (_, output_value) in training_data]):
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
