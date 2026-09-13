from __future__ import annotations

import random

from perceptron.model.association_layer import AssociationLayer
from perceptron.model.state_layer import StateLayer


class LinearClassifierNetwork:
    def __init__(
        self,
        cardinality: int,
        dimension: int,
        input_bounds: list[tuple[float, float]],
        required_active: int | None = None,
    ) -> None:

        assert cardinality >= 1
        self.cardinality = cardinality

        self.dimension = dimension

        assert len(input_bounds) == dimension
        self.input_bounds = input_bounds

        # how many of the cardinality hidden nodes must be active for the output to fire -
        # defaults to cardinality (AND, i.e. every hidden node must agree). 1 gives OR (any
        # one hidden node is enough); anything in between gives a general k-of-n gate, in
        # the spirit of a MADALINE-style committee machine.
        required_active = cardinality if required_active is None else required_active
        assert 1 <= required_active <= cardinality
        self.required_active = required_active

        self.input_layer = StateLayer(dimension, input_bounds)

        self.hidden_layer = AssociationLayer(size=cardinality, input_layer=self.input_layer)

        # the output layer is a single neuron, fully connected to the hidden layer, that
        # activates once at least required_active of its input nodes are active
        self.output_layer = AssociationLayer(size=1, input_layer=self.hidden_layer)
        output_node = self.output_layer.nodes[0]
        output_node.update_input_weights([1.0 for _ in self.hidden_layer.nodes])
        output_node.threshold = -float(self.required_active - 1)

    def update_state_layer(self, x_: tuple[float, ...]) -> None:
        self.input_layer.update_state(x_)

    def classify_state(self, state: tuple[float, ...]) -> float:
        self.update_state_layer(state)
        return self.output_layer.nodes[0].value()

    def learn(self, learning_rate: float, state: tuple[float, ...], category: float) -> None:

        self.update_state_layer(state)

        output = self.output_layer.nodes[0].value()
        if output == category:
            return

        if category == 1:
            # false negative: too few hidden nodes are active - flipping any inactive one
            # to active can only increase the count, moving the output toward firing.
            candidates = [node for node in self.hidden_layer.nodes if node.value() == 0.0]
        else:
            # false positive: enough hidden nodes are active to fire - flipping any active
            # one to inactive can only decrease the count, moving the output toward silent.
            # This (and the branch above) holds for any required_active, not just AND -
            # the output is a monotonically non-decreasing function of how many hidden
            # nodes are active, regardless of the specific threshold.
            candidates = [node for node in self.hidden_layer.nodes if node.value() == 1.0]

        responsible_node = min(candidates, key=lambda node: abs(node.z()))
        responsible_node.learn(learning_rate, category)

    def randomize(self) -> None:
        for node in self.hidden_layer.nodes:
            node.update_input_weights([random.uniform(-2, 2) for _ in self.input_layer.nodes])
            node.threshold = random.uniform(-5, 5)

    @classmethod
    def randomized(
        cls,
        cardinality: int,
        dimension: int,
        input_bounds: list[tuple[float, float]],
        required_active: int | None = None,
    ) -> LinearClassifierNetwork:
        network = cls(cardinality, dimension, input_bounds, required_active)
        network.randomize()
        return network
