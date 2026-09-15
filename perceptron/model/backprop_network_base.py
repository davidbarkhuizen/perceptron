from __future__ import annotations

from perceptron.model.backprop_layer import BackpropLayer
from perceptron.model.state_layer import StateLayer


class BackpropNetworkBase:
    """
    Shared machinery behind BackpropClassifierNetwork and MultiClassBackpropClassifierNetwork:
    layer assembly (input -> hidden layer(s) -> output layer), the forward pass, gradient
    application, the hidden-layer half of backprop, and snapshot/restore. The two subclasses
    differ only in output-layer size/shape (a single node vs class_count nodes), the resulting
    predict_*/classify_state contract, and randomize()'s initialization scheme (see each
    subclass's own docstring) - genuinely different concerns, not duplicated ones, so they stay
    out of this base.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        output_size: int,
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

        self.output_layer = BackpropLayer(size=output_size, input_layer=previous_layer)

        # drives both the backward pass and snapshot/restore uniformly - every layer whose
        # weights/bias are actually trained, in forward order
        self.trainable_layers: list[BackpropLayer] = self.hidden_layers + [self.output_layer]

    def update_state_layer(self, state: tuple[float, ...]) -> None:
        self.input_layer.update_state(state)

    def _forward_outputs(self, state: tuple[float, ...]) -> list[float]:
        self.update_state_layer(state)
        for layer in self.trainable_layers:
            layer.forward()
        return [node.value() for node in self.output_layer.nodes]

    def _backward_hidden_layers(self) -> None:
        for layer_index in reversed(range(len(self.hidden_layers))):
            next_layer = self.trainable_layers[layer_index + 1]
            for own_index, node in enumerate(self.hidden_layers[layer_index].nodes):
                node.compute_hidden_delta(next_layer.nodes, own_index)

    def _apply_gradients(self, learning_rate: float) -> None:
        for layer in self.trainable_layers:
            for node in layer.nodes:
                node.apply_gradient(learning_rate)

    def snapshot(self) -> list[list[tuple[list[float], float]]]:
        return [
            [(list(node.input_node_weights), node.bias) for node in layer.nodes] for layer in self.trainable_layers
        ]

    def restore(self, snapshot: list[list[tuple[list[float], float]]]) -> None:
        for layer, layer_snapshot in zip(self.trainable_layers, snapshot):
            for node, (weights, bias) in zip(layer.nodes, layer_snapshot):
                node.update_input_weights(weights)
                node.bias = bias
