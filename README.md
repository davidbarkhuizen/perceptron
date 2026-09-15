# first-principles-networks

A small, dependency-light implementation of two classifier families, built from first
principles. `LinearClassifierNetwork` composes state (input) nodes and association
(weighted, thresholded) nodes/layers per Rosenblatt's perceptron (1958), trained with the
classic perceptron learning rule (and a MADALINE-style minimum-disturbance rule once more
than one hidden node is used). `BackpropClassifierNetwork` is a sigmoid, gradient-descent
network of arbitrary depth, added alongside it - a genuinely different learning rule, not a
retrofit - so it can represent targets (like XOR) the discrete model structurally can't.

## docs

- [structure](docs/structure.md) — module layout, and how each network composes
- [setup](docs/setup.md) — requirements, install, and running the tests
- [demos](docs/demos.md) — the demo scripts and what each one shows
- [theory](docs/theory.md) — Rosenblatt's perceptron theory, and reference material
- [research and analysis](docs/research-and-analysis.md) — investigations behind a design
  decision, with the measurements that drove it
- [vectorization](docs/vectorization.md) — overview of a 3-part workplan (new array-based model
  classes, the numpy interface they need, and a hand-built Rust core implementing it), should
  this repo ever move away from pure Python
- [mini-batch gradient descent](docs/mini-batch-gradient-descent.md) — workplan for batching
  gradient updates, to unblock a re-test of momentum under lower-noise gradients
- [convolutional layers](docs/convolutional-layers.md) — workplan for a from-scratch conv
  layer, local receptive fields and weight sharing, on the existing image datasets
