from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    The vectorized counterpart to backprop_node.sigmoid: the same 1/(1+e^-z) formula, but relying
    on numpy's own overflow behavior instead of a try/except - large negative z drives
    np.exp(-z) to inf, and 1/(1+inf) is 0.0 under IEEE 754, which is exactly the limiting value
    backprop_node.sigmoid's OverflowError branch returns by hand. Confirmed by
    tests/test_array_layer.py's dedicated overflow-boundary sweep, not assumed from the formulas
    looking equivalent - see docs/vectorized-array-classes.md's "numerical parity validation".
    """

    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


class ArrayLayer:
    """
    One backprop layer's weights/activations as whole arrays, not `size` separate BackpropNode
    objects - see docs/vectorized-array-classes.md's "class design" section for the full
    forward/backward/gradient formula table this class implements incrementally, stage by stage.
    Built incrementally: single-example forward() first, then this stage's forward_batch().
    """

    def __init__(self, size: int, input_size: int) -> None:
        self.size = size
        self.input_size = input_size

        self.W: np.ndarray = np.zeros((size, input_size))
        self.b: np.ndarray = np.zeros(size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        self.a = sigmoid(self.z)
        return self.a

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X.shape == (batch_size, input_size); numpy broadcasts + self.b across every row, the
        # same formula as forward() applied to a whole batch at once instead of once per example
        self.Z = X @ self.W.T + self.b
        self.A = sigmoid(self.Z)
        return self.A

    def compute_output_delta(self, reference: np.ndarray) -> None:
        self.delta = (self.a - reference) * self.a * (1.0 - self.a)

    def compute_hidden_delta(self, next_layer: "ArrayLayer") -> None:
        # next_layer.W.T @ next_layer.delta replaces BackpropNode.compute_hidden_delta's
        # per-node Python sum() over downstream nodes with one matmul for the whole layer
        downstream = next_layer.W.T @ next_layer.delta
        self.delta = downstream * self.a * (1.0 - self.a)

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.delta_batch = (self.A - reference_batch) * self.A * (1.0 - self.A)

    def compute_hidden_delta_batch(self, next_layer: "ArrayLayer") -> None:
        # next_layer.delta_batch.shape == (batch_size, next_layer.size); next_layer.W.shape ==
        # (next_layer.size, self.size), so next_layer.delta_batch @ next_layer.W stacks
        # compute_hidden_delta's single-example downstream computation over every batch row
        downstream = next_layer.delta_batch @ next_layer.W
        self.delta_batch = downstream * self.A * (1.0 - self.A)
