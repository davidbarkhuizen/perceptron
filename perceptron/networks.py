

from perceptron.primitives import AssociationLayer, AssociationNode, StateLayer, StateNode
import random

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

        self.state_layer = StateLayer(dimension, input_bounds) 

        self.first_association_layer = AssociationLayer(
            nodes = [AssociationNode(
                parent_nodes=[node for node in self.state_layer.nodes],
                parent_node_weights=[random.uniform(-2, 2) for node in self.state_layer.nodes],
                threshold=random.uniform(0, 3)
            ) for i in range(cardinality)],
            parent_layer=self.state_layer
        )

        # output a_layer consists of a 
        # - a single neuron
        #   * fully connected to all its parents
        #   * that activates when all its parent nodes are active
        # return sum([self.parent_nodes[i].evaluate() * self.parent_node_weights[i]

        self.final_association_layer = AssociationLayer(   
            nodes = [AssociationNode(
                threshold=float(cardinality - 2),
                parent_nodes=[node for node in self.first_association_layer.nodes],
                parent_node_weights=[1.0 for _ in self.first_association_layer.nodes]
            )],
        )

    def update_state_layer(self, x_: tuple[float]) -> None:
        self.state_layer.update_state(x_)

    def classify_state(self, state: tuple[float]) -> int:
        self.update_state_layer(state)
        return self.final_association_layer.nodes[0].evaluate()


