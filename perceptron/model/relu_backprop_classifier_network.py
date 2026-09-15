from __future__ import annotations

from perceptron.model.backprop_classifier_network import BackpropClassifierNetwork
from perceptron.model.relu_layer import ReLULayer


class ReLUBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    A ReLU-hidden-layer sibling of BackpropClassifierNetwork, using ReLU (see ReLULayer) instead
    of sigmoid for every hidden layer - the output layer stays exactly as BackpropClassifierNetwork
    built it (sigmoid, quadratic loss), since ReLU is a hidden-layer-only convention (see
    ReLUNode's own docstring for why). Structurally this is a single class-attribute override
    (hidden_layer_cls = ReLULayer): every other method (learn/_backward/randomize/randomized/
    snapshot/restore) is inherited unchanged and stays correct, since neither the backward-pass
    plumbing nor the weight-update rule cares which activation a hidden node used - only
    ReLUNode's own forward()/compute_hidden_delta differ from BackpropNode's.

    Deliberately isolates the activation-function question from everything else already measured
    in this codebase's sibling classes: this class keeps BackpropClassifierNetwork's own
    (unmodified) randomize() scheme and quadratic output loss, so a real measurement against it
    tests ReLU specifically, not ReLU bundled with an init-scheme or loss-function change at the
    same time. See docs/research-and-analysis.md's "ReLU hidden-layer activation" entry for the
    measured comparison against the sigmoid-hidden-layer baseline.

    Deliberately not a replacement for BackpropClassifierNetwork, and not (yet) used by any
    existing demo - the same additive, keep-both pattern as every other sibling class in this
    codebase.
    """

    hidden_layer_cls = ReLULayer
