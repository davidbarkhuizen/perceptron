from __future__ import annotations

import random

from perceptron.model.backprop_layer import BackpropLayer
from perceptron.model.state_layer import StateLayer


class BackpropClassifierNetwork:
    """
    A sigmoid-activation, gradient-descent-trained network of arbitrary depth
    (input -> hidden layer(s) -> a trainable single-node output layer), added alongside
    LinearClassifierNetwork rather than as a retrofit of it: AssociationNode's hard step
    function and discrete minimum-disturbance update rule are fundamentally different from
    gradient-based learning, and every hidden node here can contribute either positively or
    negatively to the output (unlike LinearClassifierNetwork's fixed weight-of-1.0-per-node
    output layer, which can only ever be a monotonically non-decreasing function of how many
    hidden nodes are active) - see demo_nonrepresentable_target.py for what that restriction
    can't express, and demo_backprop_xor.py for this class succeeding on exactly that target.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
    ) -> None:

        assert len(layer_sizes) >= 1, "layer_sizes must specify at least one hidden layer"
        assert all(size >= 1 for size in layer_sizes), f"every hidden layer must have at least 1 node; got {layer_sizes}"

        self.dimension = dimension

        assert len(input_bounds) == dimension
        assert all(hi > lo for lo, hi in input_bounds), f"input_bounds must all have positive width; got {input_bounds}"
        self.input_bounds = input_bounds

        self.input_layer = StateLayer(dimension, input_bounds)

        self.hidden_layers: list[BackpropLayer] = []
        previous_layer: StateLayer | BackpropLayer = self.input_layer
        for size in layer_sizes:
            layer = BackpropLayer(size=size, input_layer=previous_layer)
            self.hidden_layers.append(layer)
            previous_layer = layer

        self.output_layer = BackpropLayer(size=1, input_layer=previous_layer)

        # drives both the backward pass and snapshot/restore uniformly - every layer whose
        # weights/bias are actually trained, in forward order
        self.trainable_layers: list[BackpropLayer] = self.hidden_layers + [self.output_layer]

    def update_state_layer(self, state: tuple[float, ...]) -> None:
        self.input_layer.update_state(state)

    def _forward(self, state: tuple[float, ...]) -> float:
        self.update_state_layer(state)
        for layer in self.trainable_layers:
            layer.forward()
        return self.output_layer.nodes[0].value()

    def predict_probability(self, state: tuple[float, ...]) -> float:
        return self._forward(state)

    def classify_state(self, state: tuple[float, ...]) -> float:
        return 1.0 if self.predict_probability(state) > 0.5 else 0.0

    def learn(self, learning_rate: float, state: tuple[float, ...], category: float) -> None:
        self._forward(state)
        self._backward(category)
        self._apply_gradients(learning_rate)

    def _backward(self, reference_value: float) -> None:
        self.output_layer.nodes[0].compute_output_delta(reference_value)

        for layer_index in reversed(range(len(self.hidden_layers))):
            next_layer = self.trainable_layers[layer_index + 1]
            for own_index, node in enumerate(self.hidden_layers[layer_index].nodes):
                node.compute_hidden_delta(next_layer.nodes, own_index)

    def _apply_gradients(self, learning_rate: float) -> None:
        for layer in self.trainable_layers:
            for node in layer.nodes:
                node.apply_gradient(learning_rate)

    def half_widths(self) -> list[float]:
        return [(hi - lo) / 2.0 for lo, hi in self.input_bounds]

    def randomize(self) -> None:
        # first hidden layer: same scale-invariance rationale as
        # LinearClassifierNetwork.randomize() - each weight's range scales inversely with its
        # own input dimension's half-width, so a random node has a similar chance of splitting
        # the input space regardless of input_bounds' scale. Every later trainable layer's
        # inputs are already sigmoid-normalised to (0, 1), so a fixed small range suffices.
        # Independent random draws per node (not a shared default) are required here, unlike
        # AssociationNode's identical default weights being harmless - identical starting
        # weights across nodes in a BackpropLayer would receive identical gradients forever
        # and the layer would collapse to one effective unit.
        half_widths = self.half_widths()
        for node in self.hidden_layers[0].nodes:
            node.update_input_weights(
                [random.uniform(-2.0 / half_width, 2.0 / half_width) for half_width in half_widths]
            )
            node.bias = random.uniform(-1.0, 1.0)

        for layer in self.trainable_layers[1:]:
            for node in layer.nodes:
                node.update_input_weights([random.uniform(-1.0, 1.0) for _ in node.input_nodes])
                node.bias = random.uniform(-1.0, 1.0)

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
    ) -> "BackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, input_bounds)
        network.randomize()
        return network

    def snapshot(self) -> list[list[tuple[list[float], float]]]:
        return [
            [(list(node.input_node_weights), node.bias) for node in layer.nodes] for layer in self.trainable_layers
        ]

    def restore(self, snapshot: list[list[tuple[list[float], float]]]) -> None:
        for layer, layer_snapshot in zip(self.trainable_layers, snapshot):
            for node, (weights, bias) in zip(layer.nodes, layer_snapshot):
                node.update_input_weights(weights)
                node.bias = bias
