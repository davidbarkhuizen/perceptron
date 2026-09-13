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
    linear_classifier_network.py  input -> hidden (association) -> output (AND) layers
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
    demo.py                       standalone script that trains a classifier and plots the result
    demo_cardinality_sweep.py     trains classifiers at several cardinalities and compares convergence
    demo_unreachable_class.py     shows random_alternating_training_data's max_attempts guard tripping
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
- a single-node **output layer** that ANDs the hidden layer's activations (via weights of 1
  and a threshold of `-(cardinality - 1)`).

Each `AssociationNode` updates via the perceptron learning rule
(`w += learning_rate * (reference - actual) * input`), and `train.py` drives this over a
shuffled, class-balanced set of training examples generated from a reference classifier. On
a misclassification, `LinearClassifierNetwork.learn()` doesn't update every hidden node —
it picks the single node closest to flipping (smallest `|z()|`) among those responsible for
the error, a minimum-disturbance rule in the spirit of Widrow's MADALINE, needed once
`cardinality > 1` so hidden nodes can specialize into different half-planes instead of all
converging to the same one. Unlike the single-neuron case (see [theory](theory.md)), this
has no convergence guarantee analogous to Rosenblatt's theorem — there's no proof it finds a
matching set of hyperplanes in finite steps, or at all, for an arbitrary target polytope. In
practice it converges well for modest cardinality (see `tests/test_training_pipeline.py`'s
cardinality=2 case), but that's an empirical observation, not a theorem.
