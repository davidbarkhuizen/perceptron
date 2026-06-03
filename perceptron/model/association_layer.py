from __future__ import annotations

from perceptron.model.association_node import AssociationNode
from perceptron.model.state_layer import StateLayer


class AssociationLayer:
    """
    association layers
    """

    def __init__(self, size: int, input_layer: StateLayer | AssociationLayer) -> None:

        self.size: int = size

        self.input_layer: StateLayer | AssociationLayer = input_layer

        self.nodes: list[AssociationNode] = [AssociationNode(input_nodes=self.input_layer.nodes) for _ in range(size)]
