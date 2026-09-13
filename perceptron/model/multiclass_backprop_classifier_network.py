from __future__ import annotations

import json
import math
import random

from perceptron.model.backprop_layer import BackpropLayer
from perceptron.model.state_layer import StateLayer


class MultiClassBackpropClassifierNetwork:
    """
    A one-vs-rest multi-class sibling of BackpropClassifierNetwork, built entirely on the same
    BackpropNode/BackpropLayer building blocks - a new class, not a retrofit, because
    classify_state()'s return type/contract changes (a class index, not a 0.0/1.0 float), and
    every existing backprop demo depends on the binary float contract.

    The output layer has class_count nodes instead of one; each is trained independently
    against a one-hot target (BackpropNode.compute_output_delta needs no changes for this - it
    was already a per-node, sibling-independent computation). At inference, the predicted class
    is whichever output node has the highest activation.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
    ) -> None:

        assert len(layer_sizes) >= 1, "layer_sizes must specify at least one hidden layer"
        assert all(size >= 1 for size in layer_sizes), f"every hidden layer must have at least 1 node; got {layer_sizes}"
        assert class_count >= 2, f"class_count must be at least 2; got {class_count}"

        self.dimension = dimension

        assert len(input_bounds) == dimension
        assert all(hi > lo for lo, hi in input_bounds), f"input_bounds must all have positive width; got {input_bounds}"
        self.input_bounds = input_bounds

        self.class_count = class_count

        self.input_layer = StateLayer(dimension, input_bounds)

        self.hidden_layers: list[BackpropLayer] = []
        previous_layer: StateLayer | BackpropLayer = self.input_layer
        for size in layer_sizes:
            layer = BackpropLayer(size=size, input_layer=previous_layer)
            self.hidden_layers.append(layer)
            previous_layer = layer

        self.output_layer = BackpropLayer(size=class_count, input_layer=previous_layer)

        self.trainable_layers: list[BackpropLayer] = self.hidden_layers + [self.output_layer]

    def update_state_layer(self, state: tuple[float, ...]) -> None:
        self.input_layer.update_state(state)

    def _forward(self, state: tuple[float, ...]) -> list[float]:
        self.update_state_layer(state)
        for layer in self.trainable_layers:
            layer.forward()
        return [node.value() for node in self.output_layer.nodes]

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return self._forward(state)

    def classify_state(self, state: tuple[float, ...]) -> int:
        probabilities = self.predict_probabilities(state)
        return max(range(self.class_count), key=lambda i: probabilities[i])

    def learn(self, learning_rate: float, state: tuple[float, ...], category: int) -> None:
        self._forward(state)
        self._backward(category)
        self._apply_gradients(learning_rate)

    def _backward(self, category: int) -> None:
        for i, node in enumerate(self.output_layer.nodes):
            node.compute_output_delta(1.0 if i == category else 0.0)

        for layer_index in reversed(range(len(self.hidden_layers))):
            next_layer = self.trainable_layers[layer_index + 1]
            for own_index, node in enumerate(self.hidden_layers[layer_index].nodes):
                node.compute_hidden_delta(next_layer.nodes, own_index)

    def _apply_gradients(self, learning_rate: float) -> None:
        for layer in self.trainable_layers:
            for node in layer.nodes:
                node.apply_gradient(learning_rate)

    def randomize(self) -> None:
        # fan-in-aware initialization, unlike BackpropClassifierNetwork.randomize()'s
        # per-dimension-bounds-width scaling (tuned for 1-2D geometric problems) - that scaling
        # produces exploding pre-activation sums (guaranteed sigmoid saturation) once fan-in
        # reaches the tens or hundreds, as it does here (dimension=64 for 8x8 digit images).
        # Validated empirically against the real bundled digits dataset before this class was
        # written: 99.5% training accuracy, 96.9% test accuracy.
        previous_size = self.dimension
        for layer in self.trainable_layers:
            limit = 1.0 / math.sqrt(previous_size)
            for node in layer.nodes:
                node.update_input_weights([random.uniform(-limit, limit) for _ in range(previous_size)])
                node.bias = random.uniform(-limit, limit)
            previous_size = layer.size

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
    ) -> "MultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, input_bounds, class_count)
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

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "layer_sizes": [layer.size for layer in self.hidden_layers],
                    "dimension": self.dimension,
                    "input_bounds": self.input_bounds,
                    "class_count": self.class_count,
                    "snapshot": self.snapshot(),
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "MultiClassBackpropClassifierNetwork":
        with open(path) as f:
            state = json.load(f)

        network = cls(
            state["layer_sizes"],
            state["dimension"],
            [tuple(bound) for bound in state["input_bounds"]],
            state["class_count"],
        )
        network.restore(state["snapshot"])
        return network
