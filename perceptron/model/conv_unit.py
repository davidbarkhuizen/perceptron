from __future__ import annotations

from typing import Sequence

from perceptron.model.base_node import AbstractNode
from perceptron.model.conv_kernel import ConvKernel


class ConvUnit(AbstractNode):
    """
    One convolutional output spatial position within one channel - composition, not inheritance
    from BackpropNode (see docs/convolutional-layers.md's "the architectural point that matters
    more than any single function" for why: BackpropNode's input_node_weights is an owned,
    rebindable instance attribute, which fights a weight list many units need to read
    identically rather than accommodating it). Only inherits from AbstractNode, a pure marker
    interface with no weight-related state to conflict with, so a ConvUnit is a valid
    input_nodes entry for a downstream dense BackpropLayer with zero special-casing there.

    forward()/compute_hidden_delta() match BackpropNode/ReLUNode's own formulas exactly (ReLU
    activation - relu_layer.py - the standard default for convolutional hidden layers), reading
    weights from a shared ConvKernel instead of an owned list.
    """

    def __init__(self, input_nodes: Sequence[AbstractNode], kernel: ConvKernel) -> None:

        assert len(input_nodes) == len(kernel.weights), (
            f"receptive field size ({len(input_nodes)}) must match the kernel's own weight "
            f"count ({len(kernel.weights)})"
        )

        self.input_nodes: Sequence[AbstractNode] = input_nodes
        self.kernel = kernel

        # populated by forward()/compute_hidden_delta() - see BackpropNode's own identical
        # convention (backprop_node.py) for why these have no default
        self._activation: float
        self.delta: float

    def z(self) -> float:
        return (
            sum(node.value() * weight for node, weight in zip(self.input_nodes, self.kernel.weights))
            + self.kernel.bias
        )

    def forward(self) -> float:
        self._activation = max(0.0, self.z())
        return self._activation

    def value(self) -> float:
        return self._activation

    def compute_output_delta(self, reference_value: float) -> None:
        raise NotImplementedError(
            "ConvUnit is a hidden-layer unit, not an output one - see ReLUNode's identical "
            "guard (relu_layer.py), which this mirrors: an unbounded ReLU activation isn't "
            "suited to any of this codebase's output-layer contracts."
        )

    def compute_hidden_delta(self, next_layer_nodes: Sequence["ConvUnit"], own_index: int) -> None:
        downstream = sum(node.delta * node.input_node_weights[own_index] for node in next_layer_nodes)
        self.delta = downstream if self.value() > 0.0 else 0.0

    def accumulate_gradient(self) -> None:
        receptive_field_values = [node.value() for node in self.input_nodes]
        self.kernel.accumulate_gradient(self.delta, receptive_field_values)
