import random

from perceptron.model.association_layer import AssociationLayer
from perceptron.model.state_layer import StateLayer


class LinearClassifierNetwork:
    def __init__(self, cardinality: int, dimension: int, input_bounds: list[tuple[float, float]]) -> None:

        self.cardinality = cardinality

        self.dimension = dimension

        assert len(input_bounds) == dimension
        self.input_bounds = input_bounds

        self.input_layer = StateLayer(dimension, input_bounds)

        self.hidden_layer = AssociationLayer(size=cardinality, input_layer=self.input_layer)

        # the output layer is a single neuron, fully connected to the hidden layer,
        # that activates (ANDs) when all of its input nodes are active
        self.output_layer = AssociationLayer(size=1, input_layer=self.hidden_layer)
        output_node = self.output_layer.nodes[0]
        output_node.update_input_weights([1.0 for _ in self.hidden_layer.nodes])
        output_node.threshold = -float(self.hidden_layer.size - 1)

    def update_state_layer(self, x_: tuple[float, ...]) -> None:
        self.input_layer.update_state(x_)

    def classify_state(self, state: tuple[float, ...]) -> float:
        self.update_state_layer(state)
        return self.output_layer.nodes[0].value()

    def learn(self, learning_rate: float, state: tuple[float, ...], category: int) -> None:

        self.update_state_layer(state)

        output = self.output_layer.nodes[0].value()
        if output == category:
            return

        if category == 1:
            # false negative: AND needs every hidden node active; at least one isn't.
            candidates = [node for node in self.hidden_layer.nodes if node.value() == 0.0]
        else:
            # false positive: AND fired, so every hidden node is currently active.
            candidates = [node for node in self.hidden_layer.nodes if node.value() == 1.0]

        responsible_node = min(candidates, key=lambda node: abs(node.z()))
        responsible_node.learn(learning_rate, category)

    def randomize(self) -> None:
        for node in self.hidden_layer.nodes:
            node.update_input_weights([random.uniform(-2, 2) for _ in self.input_layer.nodes])
            node.threshold = random.uniform(-5, 5)
