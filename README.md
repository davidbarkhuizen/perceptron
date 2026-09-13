# perceptron

A small, dependency-light implementation of Rosenblatt's perceptron (1958), built from
first principles: state (input) nodes, association (weighted, thresholded) nodes and
layers, wired into a single-hidden-layer linear classifier network, trained with the
classic perceptron learning rule.

See [`docs/theory.md`](docs/theory.md) for the underlying theory and references.

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
                                   training-data plotting
  train.py                        random training-data generation and the training loop
  demo.py                         standalone script that trains a classifier and plots the result
  demo_cardinality_sweep.py       trains classifiers at several cardinalities and compares convergence
  demo_unreachable_class.py       shows random_alternating_training_data's max_attempts guard tripping
tests/
  test_networks.py                exercises training-data generation and training convergence;
                                   headless (matplotlib `Agg` backend, no windows shown)
docs/
  theory.md                       Rosenblatt perceptron theory and reference material
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
converging to the same one.

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

Runs `tests/test_networks.py` under pytest. The tests are headless: they build the same
convergence-curve and decision-boundary charts as `. cli demo`, using matplotlib's `Agg`
backend, but never open a window — so they run unattended, e.g. in CI.

`. cli clean` removes `__pycache__`/`.pytest_cache` directories (used automatically before
`. cli test`).

## demo

    . cli demo

Runs `perceptron/demo.py`, which replicates `test_training_of_linear_classifier` outside
of pytest: trains a classifier against a random reference classifier and pops up the same
convergence curve and decision-boundary charts. Unlike the test, it isn't time-boxed — the
windows stay open until you close them. It uses `cardinality=2` (two hyperplanes ANDed
together) to also showcase the minimum-disturbance multi-unit learning rule described above.
It also prints the trained student's classification of a fresh point (never seen during
training) alongside the reference's, to show the trained classifier actually being used to
predict, not just compared to the reference by eye on a chart.

## demo: cardinality sweep

    . cli demo-cardinality-sweep

Runs `perceptron/demo_cardinality_sweep.py`, which trains independent reference/student
pairs at `cardinality = 1, 2, 3, 4` and overlays their disagreement-rate convergence curves
on two charts, each built from the series smoothed with a trailing moving average (to see
the trend through the sampling noise from `classification_disagreement_rate`'s small
per-checkpoint sample size) — one linear-scale, one log-scale (better for comparing how fast
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

Runs `perceptron/demo_unreachable_class.py`, a headless, console-only demo of
`random_alternating_training_data`'s safety guard: it first generates training data from a
normal, randomly initialised classifier (retrying if an unlucky `randomize()` happens to make
one class unreachable within the bounds — this alone can occasionally happen), then
deliberately constructs a classifier with tiny weights and a large threshold, whose decision
boundary never crosses its bounds, and shows the resulting `RuntimeError` being raised and
caught instead of hanging forever.
