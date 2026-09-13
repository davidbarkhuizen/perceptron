from __future__ import annotations

from typing import Sequence

from perceptron.model.backprop_node import BackpropNode
from perceptron.model.state_layer import StateLayer


class BackpropLayer:
    """
    A layer of BackpropNodes, each fully connected to the given input layer - modeled on
    AssociationLayer, but with an explicit forward() pass: node.value() is a pure cache read,
    so the layer-level forward() is what actually populates every node's cached activation.
    """

    def __init__(self, size: int, input_layer: StateLayer | "BackpropLayer") -> None:

        assert size >= 1, f"a layer must have at least 1 node; got size={size}"

        self.size: int = size

        self.input_layer: StateLayer | "BackpropLayer" = input_layer

        self.nodes: Sequence[BackpropNode] = [BackpropNode(input_nodes=self.input_layer.nodes) for _ in range(size)]

    def forward(self) -> None:
        for node in self.nodes:
            node.forward()
