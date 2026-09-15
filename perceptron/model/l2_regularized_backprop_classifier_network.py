from __future__ import annotations

from perceptron.model.backprop_classifier_network import BackpropClassifierNetwork
from perceptron.model.l2_regularization_layer import make_l2_layer_cls


class L2RegularizedBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    An L2 (weight decay) regularized sibling of BackpropClassifierNetwork, adding
    l2_lambda*weight to every weight's gradient (see make_l2_layer_cls) - bias is never
    regularized. Structurally this sets both hidden_layer_cls and output_layer_cls (the same
    pattern MomentumBackpropClassifierNetwork uses, for the same reason: this modifies the
    weight-update rule itself, shared by every trainable layer, not just one) as instance
    attributes in __init__, before BackpropNetworkBase.__init__ runs. Every other method
    (learn/_backward/randomize/snapshot/restore) is inherited unchanged.

    l2_lambda is a required constructor parameter, not a keyword default, matching
    MomentumBackpropClassifierNetwork's own posture: see docs/research-and-analysis.md's "L2
    weight regularization" entry for what was actually measured before recommending (or not
    recommending) any particular value.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        l2_lambda: float,
    ) -> None:
        layer_cls = make_l2_layer_cls(l2_lambda)
        self.hidden_layer_cls = layer_cls
        self.output_layer_cls = layer_cls
        super().__init__(layer_sizes, dimension, input_bounds)

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        l2_lambda: float,
    ) -> "L2RegularizedBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, input_bounds, l2_lambda)
        network.randomize()
        return network
