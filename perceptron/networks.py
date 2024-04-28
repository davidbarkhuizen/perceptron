from __future__ import annotations
import random
from perceptron.primitives import AssociationLayer, AssociationNode, StateLayer

class LinearClassifierNetwork:

    def __init__(self, 
        cardinality: int, 
        dimension: int,
        input_bounds: list[tuple[float, float]]
    ) -> None:

        self.cardinality = cardinality

        self.dimension = dimension

        assert(len(input_bounds) == dimension)
        self.input_bounds = input_bounds

        self.input_layer = StateLayer(dimension, input_bounds) 

        self.hidden_layer = AssociationLayer(
            size=cardinality,
            parent_layer=self.input_layer
        )

        # output a_layer consists of a 
        # - a single neuron
        #   * fully connected to all its parents
        #   * that activates when all its parent nodes are active
        # return sum([self.parent_nodes[i].evaluate() * self.parent_node_weights[i]

        self.output_layer = AssociationLayer(size=1, parent_layer=self.hidden_layer)
        output_node = self.output_layer.nodes[0]
        output_node.parent_node_weights = [1.0 for _ in self.hidden_layer.nodes]
        output_node.threshold = - float(self.hidden_layer.size - 1) 

    def update_state_layer(self, x_: tuple[float]) -> None:
        self.input_layer.update_state(x_)

    def classify_state(self, state: tuple[float]) -> int:
        self.update_state_layer(state)
        return self.output_layer.nodes[0].value()

    def learn(self, learning_rate: float, state: tuple[float], category: int) -> None:

        self.update_state_layer(state)
        for node in self.hidden_layer.nodes:
            node.teach(learning_rate, category)

    def randomize(self):
        for node in self.hidden_layer.nodes:
            node.update_parent_weights([random.uniform(-2, 2) for _ in self.input_layer.nodes])
            node.threshold = random.uniform(-5, 5)
    
    def distance(self, other: LinearClassifierNetwork) -> float:
        return self.hidden_layer.nodes[0].normalised_distance(other.hidden_layer.nodes[0])
