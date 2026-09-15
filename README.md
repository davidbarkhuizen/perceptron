# perceptron

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
- [vectorization](docs/vectorization.md) — analysis and workplan for a hand-built, Rust-backed
  array core, should this repo ever move away from pure Python
