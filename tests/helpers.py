from perceptron.model.linear_classifier_network import LinearClassifierNetwork


def classifier_with_bounded_square_region(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # positive region is exactly the square [-1, 1] x [-1, 1]: x > -1, x < 1, y > -1, y < 1
    classifier = LinearClassifierNetwork(4, 2, bounds)
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [([1.0, 0.0], 1.0), ([-1.0, 0.0], 1.0), ([0.0, 1.0], 1.0), ([0.0, -1.0], 1.0)],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold
    return classifier


def classifier_with_tiny_bounded_region(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # positive region is exactly [-0.1, 0.1] x [-0.1, 0.1] - a 0.04 unit^2 square, a tiny
    # fraction of a square_bounds(10.0)-sized (400 unit^2) box (0.01%), for exercising the
    # tight-box positive-region sampling optimisation (see geometry.positive_region_bounding_box)
    classifier = LinearClassifierNetwork(4, 2, bounds)
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [([1.0, 0.0], 0.1), ([-1.0, 0.0], 0.1), ([0.0, 1.0], 0.1), ([0.0, -1.0], 0.1)],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold
    return classifier


def network_with_hidden_thresholds(
    dimension: int,
    bounds: list[tuple[float, float]],
    thresholds: list[float],
    required_active: int | None = None,
) -> LinearClassifierNetwork:

    # zero input weights make each node's z() equal to its threshold alone, regardless of
    # state - so the thresholds directly pick each node's active/inactive status
    network = LinearClassifierNetwork(len(thresholds), dimension, bounds, required_active)
    for node, threshold in zip(network.hidden_layer.nodes, thresholds):
        node.update_input_weights([0.0] * dimension)
        node.threshold = threshold
    return network
