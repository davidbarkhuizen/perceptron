from __future__ import annotations
from math import sqrt
from perceptron.model.state_node import StateNode

class AssociationNode:
            
    def __init__(self, 
        threshold: float = 0.0,
        inpute_nodes: list[StateNode | AssociationNode] = None, 
        input_node_weights: list[float] = None
    ) -> None:

        self.threshold = threshold

        self.input_nodes = inpute_nodes if inpute_nodes is not None else []        
        
        self.input_node_weights = input_node_weights if input_node_weights is not None else [
            1.0 for _ in self.input_nodes
        ]

    def update_input_weights(self, weights: list[float]) -> None:                
        assert(len(weights) == len(self.input_nodes))
        self.input_node_weights = weights

    def z(self) -> float:

        aggregate_input_value = sum([
            self.input_nodes[i].value() * self.input_node_weights[i]
                for i in range(len(self.input_nodes))
        ])

        return aggregate_input_value + self.threshold

    def value(self) -> int:
        return 1 if self.z() > 0.0 else 0

    def learn(self, 
        learning_rate: float, 
        reference_value: int
    ):

        current_value = self.value()
        correctly_categorised = current_value == reference_value
        if correctly_categorised:
            return

        if reference_value == 1 and current_value == 0:
            d = 1.0
        elif reference_value == 0 and current_value == 1:
            d = -1.0
        else:
            raise('wtf')

        w_p0 = self.threshold + learning_rate * d * 1.0
        w_p1 = self.input_node_weights[0] + learning_rate * d * self.input_nodes[0].value()
        w_p2 = self.input_node_weights[1] + learning_rate * d * self.input_nodes[1].value()

        self.threshold = w_p0
        self.update_input_weights([w_p1, w_p2])

    def normalised_distance(self, other: AssociationNode) -> float:

        input_node_count = len(self.input_node_weights)
        assert(input_node_count == len(other.input_node_weights))

        alpha = [
            self.threshold, 
            *[self.input_node_weights[i] for i in range(input_node_count)]
        ]

        beta = [
            other.threshold, 
            *[other.input_node_weights[i] for i in range(input_node_count)]
        ]

        def normalise(vector: list[float]) -> list[float]:            
            length = sqrt(sum([x**2 for x in vector]))
            return [x/length for x in vector]

        def distance(alpha: list[float], beta: list[float]) -> float:
            assert(len(alpha) == len(beta))
            if len(alpha) == 0:
                return ValueError('undefined')

            return sqrt(sum([(alpha[i] - beta[i])**2 for i in range(len(alpha))]))

        return distance(normalise(alpha), normalise(beta))