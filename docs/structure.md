# structure

[← back to README](../README.md)

```
perceptron/
  model/
    base_node.py                    AbstractNode — value() interface
    state_node.py                   sense point holding a scalar input value
    state_layer.py                  a vector of StateNodes (network input layer)
    association_node.py             weighted, thresholded neuron (z, activation, learning rule)
    association_layer.py            a layer of AssociationNodes over a given input layer
    linear_classifier_network.py    input -> hidden (association) -> output (k-of-n) layers
    backprop_node.py                sigmoid neuron trained by gradient descent (z, forward
                                     cache, output/hidden delta, gradient step)
    backprop_layer.py               a layer of BackpropNodes over a given input layer
    backprop_classifier_network.py  input -> hidden (backprop, any depth) -> trainable output
  graphics/
    chart.py                      matplotlib helpers: figures/axes, decision-boundary and
                                   training-data plotting, and expanding plot bounds to
                                   fit a classifier's bounded positive region
  geometry.py                     matplotlib-free geometry: intersecting a classifier's
                                   hidden-node half-planes to find its positive region, test
                                   whether that region is bounded and get a tight box around
                                   it, plus a symmetric input-bounds constructor
  train.py                        random training-data generation and the training loop
  evaluate.py                     comparing/measuring classifiers: class-balanced sampling
                                   (efficiently, using geometry.py's tight box around the
                                   positive region when available), sampling a point and
                                   classifying it with two networks, a permutation-invariant
                                   class-balanced disagreement metric, and series smoothing
  demos/
    demo.py                           standalone script that trains a classifier and plots the result
    demo_cardinality_sweep.py         trains classifiers at several cardinalities and compares convergence
    demo_unreachable_class.py         shows random_alternating_training_data's max_attempts guard tripping
    demo_nonrepresentable_target.py   trains against a target no required_active/cardinality can represent
    demo_backprop_xor.py              trains a BackpropClassifierNetwork on the same target - and converges
    demo_backprop_architecture_sweep.py  compares single- vs two-hidden-layer BackpropClassifierNetworks
                                       at matched node budgets, on a harder-than-XOR striped target
    demo_backprop_circular_target.py  trains against a circular target - a genuinely curved
                                       boundary, unlike any LinearClassifierNetwork's polygon
    demo_backprop_vs_linear.py        parity check: both models on the same easy, linearly-
                                       separable target
tests/                           one file per module under test, plus test_training_pipeline.py for
                                  end-to-end coverage; all headless (matplotlib `Agg` backend, no
                                  windows shown)
  helpers.py                     shared test fixtures (e.g. a classifier with a known, hand-built
                                  bounded positive region)
  test_model.py                  LinearClassifierNetwork construction, learning rule, randomized()
  test_geometry.py               square_bounds, is_positive_region_bounded
  test_train.py                  training-data generation, reachable_reference_and_training_data
  test_evaluate.py               sample_class_balanced_states, class_balanced_disagreement_rate,
                                  compare_on_random_point, smoothed_series
  test_chart.py                  reference_region_bounds, disagreement_axis_bounds
  test_training_pipeline.py      end-to-end training + convergence + decision-boundary plotting
                                  (mirrors what demo.py does, minus the windows)
  test_backprop_model.py         BackpropClassifierNetwork construction, hand-computed forward/
                                  backward pass, randomize() symmetry-breaking, snapshot/restore
  test_backprop_training_pipeline.py  proves train_linear_classifier_network drives a
                                  BackpropClassifierNetwork well past the linear ceiling on XOR,
                                  unchanged
cli                                setup / test / clean helper script
```

`LinearClassifierNetwork` composes:

- an **input layer** (`StateLayer`) of raw input values,
- a **hidden layer** of `cardinality` `AssociationNode`s, each fully connected to the input layer,
- a single-node **output layer** that fires once at least `required_active` of the hidden
  layer's nodes are active (via weights of 1 and a threshold of `-(required_active - 1)`).
  `required_active` defaults to `cardinality` (AND: every hidden node must agree); passing 1
  gives OR (any one is enough), and anything in between gives a general k-of-n gate, in the
  spirit of a MADALINE-style committee machine. `geometry.py`'s positive-region functions
  only support the AND case (`required_active == cardinality`) and `dimension == 2` - the
  true positive region under any other gate is a union of intersections, not a single
  intersection, and a higher dimension needs more than just the first two weights it reads -
  and those functions raise rather than silently return a wrong answer if called on one.

Each `AssociationNode` updates via the perceptron learning rule
(`w += learning_rate * (reference - actual) * input`), and `train.py` drives this over a
shuffled, class-balanced set of training examples generated from a reference classifier. On
a misclassification, `LinearClassifierNetwork.learn()` doesn't update every hidden node —
it picks the single node closest to flipping (smallest `|z()|`) among those responsible for
the error, a minimum-disturbance rule in the spirit of Widrow's MADALINE, needed once
`cardinality > 1` so hidden nodes can specialize into different half-planes instead of all
converging to the same one. This selection is combination-gate-agnostic: it only relies on
the output being a monotonically non-decreasing function of how many hidden nodes are
active, which holds for AND, OR, or any k-of-n `required_active` - so `learn()` needed no
changes to support gates other than AND.

Unlike the single-neuron case (see [theory](theory.md)), this
has no convergence guarantee analogous to Rosenblatt's theorem — there's no proof it finds a
matching set of hyperplanes in finite steps, or at all, for an arbitrary target polytope. In
practice it converges well for modest cardinality (see `tests/test_training_pipeline.py`'s
cardinality=2 case), but that's an empirical observation, not a theorem.

Because of that, `train.train_linear_classifier_network` uses a pocket-algorithm-style
"keep the best, not the latest" rule: training accuracy on `training_data` is measured after
every epoch, and the student is left at whichever epoch's weights scored best, not
necessarily the raw last epoch's - a no-op when training does converge (the best epoch is
then the last one), and a real difference when it doesn't (see
[demos](demos.md#demo-non-representable-target)). It returns a `ConvergenceSeries` - the
same list of `(iteration, disagreement_rate)` pairs it's always returned, so every existing
use keeps working unchanged, plus a `.diagnostic` (a `TrainingDiagnostic`) that answers,
without needing to eyeball a chart, whether that run `.converged` (every training example
correctly classified), `.plateaued` (the best epoch wasn't the last one - more epochs
already stopped helping), or was `.still_improving` (the last epoch was still the best one
seen, just not there yet).

`LinearClassifierNetwork.randomize()` scales each hidden node's weight range inversely with
its own dimension's `input_bounds` half-width (and leaves the threshold range fixed) so that
a random hidden node has a similar chance of splitting the input space regardless of how
large, small, or asymmetric `input_bounds` is - a fixed weight range would let the threshold
dominate at small bounds scales, making almost every random classifier permanently one
class.

`evaluate.sample_class_balanced_states` (used by both `random_alternating_training_data` and
`class_balanced_disagreement_rate`) draws positive-class points from a tight box around the
classifier's own positive region (`geometry.positive_region_bounding_box`) instead of
rejection-sampling all of `input_bounds`, when that box is computable. A classifier's positive
region can be a tiny fraction of `input_bounds` - measured empirically: often under 1% of the
box's area at cardinality 4 with the bounded-region requirement, as low as 0.02%, shrinking
further as cardinality grows - so naive full-box rejection sampling can need far more attempts
than `max_attempts` allows, wasting the whole budget on candidates a tighter box would have
made trivial. Falls back to `input_bounds` (byte-for-byte the previous behaviour) whenever no
tight box is computable - cardinality 1-2 (never bounded), a non-AND `required_active`, or a
classifier that isn't a `LinearClassifierNetwork` at all (both functions accept anything with
the same `input_bounds`/`classify_state` interface, e.g. `demo_nonrepresentable_target.py`'s
`XORTarget`).

## backprop

`BackpropClassifierNetwork` is an additive alternative to `LinearClassifierNetwork`, not a
retrofit of it - `AssociationNode`'s hard step function and discrete minimum-disturbance update
rule are fundamentally different from gradient-based learning. It composes `BackpropNode`s
(sigmoid activation, `a = 1/(1+e^-z)`) into `BackpropLayer`s of arbitrary depth
(`input -> hidden layer(s) -> a trainable single-node output layer`). Unlike
`LinearClassifierNetwork`'s output layer (fixed weights of `1.0` per hidden node, so its output
can only be a monotonically non-decreasing function of how many hidden nodes fire -
see `demo_nonrepresentable_target.py`), every layer here is trained, including the output layer,
so a hidden node can push the output either way. That's what lets it represent targets like XOR
that no `required_active`/`cardinality` combination can (see `demo_backprop_xor.py`).

Learning uses mean-squared-error, back-propagated by hand (no autodiff): at the output node,
`delta = (a - y) * a * (1-a)`; at a hidden node, `delta = (Σ downstream delta * weight) *
a * (1-a)` - the same shape at every layer, one rule applied throughout, in the same
first-principles spirit as [theory](theory.md). Each node caches its activation via an explicit
`forward()` pass (`value()` is a pure cache read) so a downstream node's multiple reads of an
upstream node don't each re-pay for a fresh sigmoid evaluation. `randomize()` must give every
node in a layer independent random weights - unlike `AssociationNode`'s harmless identical
default weights, identical starting weights here would give every node in a layer identical
gradients forever, collapsing it to one effective unit.

`train_linear_classifier_network` (see above) trains a `BackpropClassifierNetwork` completely
unchanged - it only ever calls `.learn()`, `.snapshot()`/`.restore()` (renamed from
`LinearClassifierNetwork`'s original `hidden_layer_snapshot`/`restore_hidden_layer` for exactly
this reason), and `.classify_state()`, all of which both classes implement. The two classes'
`.snapshot()` shapes differ (flat, hidden-layer-only for `LinearClassifierNetwork`; nested,
covering every trainable layer including the output layer for `BackpropClassifierNetwork`) but
the training loop only ever treats the snapshot as opaque, so this is invisible to it.
