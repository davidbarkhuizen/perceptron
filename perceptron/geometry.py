from perceptron.model.linear_classifier_network import LinearClassifierNetwork


def square_bounds(l: float, dimension: int = 2) -> list[tuple[float, float]]:
    return [(-l, l)] * dimension


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


def reference_positive_region_polygon(
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
    return bool(reference_positive_region_polygon(classifier))
