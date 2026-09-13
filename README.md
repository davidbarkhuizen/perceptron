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
    association_node.py           weighted, thresholded neuron (z, activation, learning rule, distance)
    association_layer.py          a layer of AssociationNodes over a given input layer
    linear_classifier_network.py  input -> hidden (association) -> output (AND) layers
  graphics/
    chart.py                      matplotlib helpers: figures/axes, decision-boundary and
                                   training-data plotting
  train.py                        random training-data generation and the training loop
tests/
  test_networks.py                exercises training-data generation and training convergence,
                                   rendering live matplotlib charts (interactive, not headless)
docs/
  theory.md                       Rosenblatt perceptron theory and reference material
cli                                setup / test / clean helper script
```

`LinearClassifierNetwork` composes:

- an **input layer** (`StateLayer`) of raw input values,
- a **hidden layer** of `cardinality` `AssociationNode`s, each fully connected to the input layer,
- a single-node **output layer** that ANDs the hidden layer's activations (via weights of 1
  and a threshold of `-(cardinality - 1)`).

Each `AssociationNode` learns independently via the perceptron update rule
(`w += learning_rate * (reference - actual) * input`), and `train.py` drives this over a
shuffled, class-balanced set of training examples generated from a reference classifier.

## requirements

- Python 3.10+
- `python3-tk` (for the interactive matplotlib `TkAgg` backend used by the tests)

## install

    . cli setup

This installs `python3-tk` via `apt`, creates a `.venv`, and installs the Python
dependencies (`matplotlib`, `pytest`) from `requirements.txt`.

## test

    . cli test

Runs `tests/test_networks.py` under pytest. The tests are interactive: they pop up
matplotlib windows (convergence curve, and the reference vs. trained decision boundary
over the training data) and pause execution for ~20 seconds so you can inspect them, then
prompt to repeat the run.

`. cli clean` removes `__pycache__`/`.pytest_cache` directories (used automatically before
`. cli test`).
