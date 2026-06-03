from __future__ import annotations

from math import sqrt

from perceptron.model.state_node import StateNode


class AssociationNode:
    def __init__(
        self,
        threshold: float = 0.0,
        input_nodes: list[StateNode | AssociationNode] | None = None,
        input_node_weights: list[float] | None = None,
    ) -> None:

        self.threshold: float = threshold

        self.input_nodes: list[StateNode | AssociationNode] = input_nodes if input_nodes is not None else []

        self.input_node_weights: list[float] = (
            input_node_weights if input_node_weights is not None else [1.0 for _ in self.input_nodes]
        )

    def update_input_weights(self, weights: list[float]) -> None:
        assert len(weights) == len(self.input_nodes)
        self.input_node_weights = weights

    def z(self) -> float:

        aggregate_input_value: float = sum(
            [self.input_nodes[i].value() * self.input_node_weights[i] for i in range(len(self.input_nodes))]
        )

        return aggregate_input_value + self.threshold

    def value(self) -> float:
        return 1.0 if self.z() > 0.0 else 0.0

    def learn(self, learning_rate: float, reference_value: float):

        current_value = self.value()
        correctly_categorised = current_value == reference_value
        if correctly_categorised:
            return

        if reference_value == 1 and current_value == 0:
            d = 1.0
        elif reference_value == 0 and current_value == 1:
            d = -1.0
        else:
            raise ValueError(f"reference_value = {reference_value}, current_value = {current_value}")

        w_p0 = self.threshold + learning_rate * d * 1.0
        w_p1 = self.input_node_weights[0] + learning_rate * d * self.input_nodes[0].value()
        w_p2 = self.input_node_weights[1] + learning_rate * d * self.input_nodes[1].value()

        self.threshold = w_p0
        self.update_input_weights([w_p1, w_p2])

    def normalised_distance(self, other: AssociationNode) -> float:

        input_node_count = len(self.input_node_weights)
        assert input_node_count == len(other.input_node_weights)

        alpha = [
            self.threshold,
            *[self.input_node_weights[i] for i in range(input_node_count)],
        ]

        beta = [
            other.threshold,
            *[other.input_node_weights[i] for i in range(input_node_count)],
        ]

        def normalise(vector: list[float]) -> list[float]:
            length = sqrt(sum([x**2 for x in vector]))
            return [x / length for x in vector]

        def distance(aleph: list[float], beth: list[float]) -> float:
            assert len(aleph) == len(beth)
            if len(aleph) == 0:
                raise ValueError("undefined distance, len(aleph) == 0")

            return sqrt(sum([(aleph[i] - beth[i]) ** 2 for i in range(len(aleph))]))

        return distance(normalise(alpha), normalise(beta))
