# structure

[← back to README](../README.md)

```
perceptron/
  model/
    base_node.py                  AbstractNode — value() interface
    state_node.py                 sense point holding a scalar input value
    state_layer.py                a vector of StateNodes (network input layer)
    association_node.py           weighted, thresholded neuron (z, activation, learning rule)
    association_layer.py          a layer of AssociationNodes over a given input layer
    linear_classifier_network.py  input -> hidden (association) -> output (k-of-n) layers
  graphics/
    chart.py                      matplotlib helpers: figures/axes, decision-boundary and
                                   training-data plotting, and expanding plot bounds to
                                   fit a classifier's bounded positive region
  geometry.py                     matplotlib-free geometry: intersecting a classifier's
                                   hidden-node half-planes to find its positive region and
                                   test whether that region is bounded, plus a symmetric
                                   input-bounds constructor
  train.py                        random training-data generation and the training loop
  evaluate.py                     comparing/measuring classifiers: sampling a point and
                                   classifying it with two networks, a permutation-invariant
                                   class-balanced disagreement metric, and series smoothing
  demos/
    demo.py                           standalone script that trains a classifier and plots the result
    demo_cardinality_sweep.py         trains classifiers at several cardinalities and compares convergence
    demo_unreachable_class.py         shows random_alternating_training_data's max_attempts guard tripping
    demo_nonrepresentable_target.py   trains against a target no required_active/cardinality can represent
tests/                           one file per module under test, plus test_training_pipeline.py for
                                  end-to-end coverage; all headless (matplotlib `Agg` backend, no
                                  windows shown)
  helpers.py                     shared test fixtures (e.g. a classifier with a known, hand-built
                                  bounded positive region)
  test_model.py                  LinearClassifierNetwork construction, learning rule, randomized()
  test_geometry.py               square_bounds, is_positive_region_bounded
  test_train.py                  training-data generation, reachable_reference_and_training_data
  test_evaluate.py               class_balanced_disagreement_rate, compare_on_random_point, smoothed_series
  test_chart.py                  reference_region_bounds, disagreement_axis_bounds
  test_training_pipeline.py      end-to-end training + convergence + decision-boundary plotting
                                  (mirrors what demo.py does, minus the windows)
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
  only support the AND case (`required_active == cardinality`) - the true positive region
  under any other gate is a union of intersections, not a single intersection, and those
  functions raise rather than silently return a wrong answer if called on one.

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
[demos](demos.md#demo-non-representable-target)).

`LinearClassifierNetwork.randomize()` scales each hidden node's weight range inversely with
its own dimension's `input_bounds` half-width (and leaves the threshold range fixed) so that
a random hidden node has a similar chance of splitting the input space regardless of how
large, small, or asymmetric `input_bounds` is - a fixed weight range would let the threshold
dominate at small bounds scales, making almost every random classifier permanently one
class.
