# perceptron

A small, dependency-light implementation of Rosenblatt's perceptron (1958), built from
first principles: state (input) nodes, association (weighted, thresholded) nodes and
layers, wired into a single-hidden-layer linear classifier network, trained with the
classic perceptron learning rule.

See [theory](#theory) below for the underlying theory and references.

## structure

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
converging to the same one. Unlike the single-neuron case (see [theory](#theory) below), this
has no convergence guarantee analogous to Rosenblatt's theorem — there's no proof it finds a
matching set of hyperplanes in finite steps, or at all, for an arbitrary target polytope. In
practice it converges well for modest cardinality (see `tests/test_training_pipeline.py`'s
cardinality=2 case), but that's an empirical observation, not a theorem.

## requirements

- Python 3.10+
- `python3-tk` (for the interactive matplotlib `TkAgg` backend used by `. cli demo`; the
  tests themselves are headless and don't need it)

## install

    . cli setup

This installs `python3-tk` via `apt`, creates a `.venv`, and installs the Python
dependencies (`matplotlib`, `pytest`) from `requirements.txt`.

## test

    . cli test

Runs everything under `tests/` with pytest, grouped one file per module under test (see the
structure above), plus `test_training_pipeline.py` for end-to-end coverage - it builds the
same convergence-curve and decision-boundary charts as `. cli demo`, using matplotlib's `Agg`
backend, but never opens a window — so all of it runs unattended, e.g. in CI.

`. cli clean` removes `__pycache__`/`.pytest_cache` directories (used automatically before
`. cli test`).

## demo

    . cli demo

Runs `perceptron/demos/demo.py`, which replicates `test_training_of_linear_classifier` outside
of pytest: trains a classifier against a random reference classifier and pops up three
windows — the convergence curve on a linear scale, the same curve on a log scale (better for
seeing how fast it converges, since disagreement tends to drop roughly exponentially — see
the cardinality-sweep demo below for more on this), and the decision-boundary chart. Both
convergence charts are smoothed with a trailing moving average. Unlike the test, it isn't
time-boxed — the windows stay open until you close them. It uses `cardinality=4` (four
hyperplanes ANDed together) to also showcase the minimum-disturbance multi-unit learning
rule described above; since a higher cardinality shrinks the reference's positive region,
it uses `train.reachable_reference_and_training_data` (also used by the cardinality-sweep
demo) to regenerate the reference rather than risk failing on one unlucky `randomize()`. A
2D convex region needs at least 3 half-planes to be bounded at all, so at `cardinality=4`
the demo also passes `geometry.is_positive_region_bounded` as that helper's `is_valid` filter,
rejecting (and regenerating) any reference whose positive region isn't a bounded, closed
shape — bounded regions are rare (~10% of random draws at this cardinality), but rejecting
on it is a cheap geometry check with no sampling, so a large attempt budget is still fast.
It also prints the trained student's classification of a fresh point (never seen during
training) alongside the reference's, to show the trained classifier actually being used to
predict, not just compared to the reference by eye on a chart. The decision-boundary
chart's bounds are widened as needed, via `chart.reference_region_bounds`, so the whole
bounded region stays visible instead of being cropped at the training bounds.

## demo: cardinality sweep

    . cli demo-cardinality-sweep

Runs `perceptron/demos/demo_cardinality_sweep.py`, which trains independent reference/student
pairs at `cardinality = 1, 2, 3, 4` and overlays their disagreement-rate convergence curves
on two charts, each built from the series smoothed with a trailing moving average (to see
the trend through the sampling noise from `class_balanced_disagreement_rate`'s small
per-class sample count) — one linear-scale, one log-scale (better for comparing how fast
each cardinality's disagreement rate drops, since it does so roughly exponentially) — and
printing each cardinality's before/after disagreement and a prediction-agreement check. A
disagreement rate of exactly 0.0 has no position on a log axis, so a fully-converged
cardinality's curve on that chart simply stops once it hits zero. Since a higher cardinality
shrinks the reference's positive region (the intersection of more half-planes), some
randomly generated reference
classifiers make one class unreachable within the given bounds — the demo regenerates the
reference (up to 20 times) rather than failing the whole sweep on one unlucky draw.

## demo: unreachable class

    . cli demo-unreachable-class

Runs `perceptron/demos/demo_unreachable_class.py`, a headless, console-only demo of
`random_alternating_training_data`'s safety guard: it first generates training data from a
normal, randomly initialised classifier (retrying if an unlucky `randomize()` happens to make
one class unreachable within the bounds — this alone can occasionally happen), then
deliberately constructs a classifier with tiny weights and a large threshold, whose decision
boundary never crosses its bounds, and shows the resulting `RuntimeError` being raised and
caught instead of hanging forever.

## theory

### external references

- Shree Nayar, Computer Science Dept, School of Engineering & Applied Sciences, Columbia
  University — https://fpcv.cs.columbia.edu/
- First Principles of Computer Vision Course —
  https://fpcv.cs.columbia.edu/, https://www.youtube.com/@firstprinciplesofcomputerv3258
- Perceptron | Neural Networks — https://www.youtube.com/watch?v=OFbnpY_k7js

### example configuration (NAND gate)

    w = [-2, 2]
    b = 3

### summary of perceptron theory, per Rosenblatt (1958)

Given:

- input `x_`: a vector of `n` inputs, `x1 .. xn`
- input weights `w_`: a vector, one weight per input, `w1 .. wn`
- bias `b` (activation threshold): a scalar

the activation function `f` is

    f(w_.x_) = 0  iff  w_.x_ <= -b
    f(w_.x_) = 1  iff  w_.x_ >  -b

Defining `z = w_.x_ + b`, the activation `a` is a function of `z`:

    a = f(z) = 0  iff  z <= 0
    a = f(z) = 1  iff  z >  0

i.e. the perceptron neuron's activation function is a step function.

For a single neuron, Rosenblatt's perceptron convergence theorem guarantees the update rule
above finds a separating hyperplane in a finite number of steps, provided the training data
is linearly separable. See the [structure](#structure) section above for how this composes
into `LinearClassifierNetwork` and what changes once `cardinality > 1`.
