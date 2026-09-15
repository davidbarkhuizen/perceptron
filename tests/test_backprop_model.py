import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip
from perceptron.geometry import square_bounds
from perceptron.model.backprop_classifier_network import BackpropClassifierNetwork


def test_layer_sizes_must_specify_at_least_one_hidden_layer():

    with pytest.raises(AssertionError):
        BackpropClassifierNetwork([], 2, square_bounds(10.0))


def test_every_hidden_layer_size_must_be_at_least_one():

    with pytest.raises(AssertionError):
        BackpropClassifierNetwork([4, 0], 2, square_bounds(10.0))


def test_dimension_must_match_bounds_length():

    with pytest.raises(AssertionError):
        BackpropClassifierNetwork([4], 2, [(-1.0, 1.0)])


def test_input_bounds_must_all_have_positive_width():

    with pytest.raises(AssertionError):
        BackpropClassifierNetwork([4], 2, [(-10.0, 10.0), (5.0, 5.0)])

    # inverted bounds (hi < lo) are equally nonsensical
    with pytest.raises(AssertionError):
        BackpropClassifierNetwork([4], 2, [(-10.0, 10.0), (5.0, -5.0)])


def test_predict_probability_matches_a_hand_computed_forward_pass():

    # a minimal 1D-input, single-hidden-node, single-output-node network, small enough to
    # state the whole forward pass exactly: z_h = w_h*x + b_h, a_h = sigmoid(z_h),
    # z_o = w_o*a_h + b_o, a_o = sigmoid(z_o). With w_h=0.5, b_h=0.1, w_o=0.8, b_o=-0.2,
    # x=2.0: z_h=1.1, a_h=sigmoid(1.1)=0.7502601055951177, z_o=0.8*a_h-0.2=0.4002080844760941,
    # a_o=sigmoid(z_o)=0.5987376536170401 (independently computed, not re-derived from the
    # implementation under test).
    network = BackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]
    hidden_node.update_input_weights([0.5])
    hidden_node.bias = 0.1
    output_node.update_input_weights([0.8])
    output_node.bias = -0.2

    assert network.predict_probability((2.0,)) == pytest.approx(0.5987376536170401)


def test_classify_state_thresholds_strictly_above_half():

    network = BackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]
    hidden_node.update_input_weights([0.0])
    hidden_node.bias = 0.0
    output_node.update_input_weights([0.0])
    output_node.bias = 0.0

    # z_o = 0.0 -> a_o = sigmoid(0) = 0.5 exactly - must not classify as active
    assert network.predict_probability((0.0,)) == pytest.approx(0.5)
    assert network.classify_state((0.0,)) == 0.0


def test_learn_matches_the_backprop_update_rule_by_hand():

    # pins the forward+backward arithmetic against independently hand-derived expected
    # values (same minimal network as the forward-pass test above):
    #   delta_o = (a_o - y) * a_o * (1 - a_o)
    #   delta_h = (delta_o * w_o) * a_h * (1 - a_h)
    #   w -= learning_rate * delta * <that weight's input value>;  b -= learning_rate * delta
    # starting weights w_h=0.5, b_h=0.1, w_o=0.8, b_o=-0.2; state x=2.0, category y=1.0,
    # learning_rate=0.1 - computed independently (not re-derived from the implementation under
    # test): delta_o=-0.09640363012729687, delta_h=-0.014450509251916271, giving
    # new_w_h=0.5028901018503833, new_b_h=0.10144505092519163, new_w_o=0.8072327797719059,
    # new_b_o=-0.19035963698727032.
    network = BackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]
    hidden_node.update_input_weights([0.5])
    hidden_node.bias = 0.1
    output_node.update_input_weights([0.8])
    output_node.bias = -0.2

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5028901018503833)
    assert hidden_node.bias == pytest.approx(0.10144505092519163)
    assert output_node.input_node_weights[0] == pytest.approx(0.8072327797719059)
    assert output_node.bias == pytest.approx(-0.19035963698727032)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = BackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0))
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
