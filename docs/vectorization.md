# vectorization

[← back to README](../README.md)

Analysis and workplan only - not a decision to build anything here, and nothing in
`perceptron/` changes as a result of this document. `docs/structure.md`'s "possible next
steps" flags NumPy vectorization as a values question ("no ML framework dependency,
everything hand-built" is this repo's own stated identity) rather than a recommendation; this
document is the detailed analysis behind that flag, worked all the way through to a concrete
workplan for one specific way it could be done - a hand-built, tightly-scoped array core,
in Rust, wrapped for Python - rather than adopting the real `numpy` package outright.

## why not just adopt real NumPy

Installing `numpy` would obviously work, and would be the pragmatic choice for anyone whose
only goal is faster training. This document exists because this repo's own identity treats
even a "just fast math" dependency as a deliberate choice, not a default - the same posture
that's kept scikit-learn as a one-time, offline, non-runtime extraction tool
(`digits_data.py`) rather than a real dependency, and pyarrow as a one-time conversion step
(`mnist_data.convert_parquet_to_binary`) rather than something training itself ever imports.
If a future maintainer decides plain `numpy` is the right call, this entire document - and the
Rust alternative it plans - is moot. It's written up because the alternative was asked to be
scoped in real detail, not because hand-rolling it is being recommended over the obvious
off-the-shelf option.

## the architectural point that matters more than any single function

Nearly every mapping below only pays off if a whole `BackpropLayer`'s weights are held as one
2D array and its activations as one 1D (or batched 2D) array, computed with one matrix
operation per layer per (batch of) example(s) - not once per node, the way every class in this
codebase currently works (`BackpropNode`, `StateNode` are individual Python objects; a layer's
"weight matrix" is really `size` separate `list[float]`s, one per node, see
`perceptron/model/base_node.py`'s `WeightedInputNode.z()`). Calling a vectorized dot product
between two 16-element arrays, once per node, inside the current per-node Python loop, would
remove only the innermost `sum()` while keeping nearly all the actual overhead (Python-level
looping, per-node object dispatch, attribute lookups) - a real but partial win, not the
"vectorized" one this exercise is about. The real payoff needs `BackpropLayer`'s internal
representation rewritten around whole-layer arrays, which is a different shape of change than
every existing sibling in this codebase (momentum, L2, ReLU, softmax - see
[structure](structure.md#backprop-siblings)) - those all override a per-node method's *body*;
this would need to change what a node's state even *is*, structurally. See "what stays out of
scope" below for how this workplan handles that.

## function-level inventory

Grounded in the actual current implementation (`perceptron/model/base_node.py`,
`backprop_node.py`, `backprop_layer.py`, `softmax_output_layer.py`, `relu_layer.py`,
`momentum_layer.py`, `l2_regularization_layer.py`, `mnist_data.py`), not a generic
vectorization checklist.

### forward pass

| current code | vectorized replacement |
|---|---|
| `WeightedInputNode.z()`: a Python `sum()` over `input_nodes[i].value() * input_node_weights[i]`, one node at a time | `z = W @ x + b` (single example) or `Z = X @ W.T + b` (batch) - one matrix-vector/matrix-matrix multiply plus a broadcasted bias add, replacing the entire per-node Python loop |
| `sigmoid(z)`: `1/(1+math.exp(-z))`, guarded by `try/except OverflowError` | elementwise `1.0 / (1.0 + exp(-z))` across a whole layer/batch at once |
| `ReLUNode.forward()`: `max(0.0, self.z())` | elementwise `max(0.0, z)` |
| `SoftmaxOutputLayer.forward()`: `max(z_values)`, list-comprehension `math.exp(z - max_z)`, `sum(exp_values)`, per-node divide | `m = max(z, axis=-1); e = exp(z - m); softmax = e / sum(e, axis=-1)` - the same numerically-stable formula, computed layer-wide in one pass instead of a Python list comprehension |

### backward pass (deltas)

| current code | vectorized replacement |
|---|---|
| `compute_output_delta` (quadratic): `(a - y) * a * (1 - a)`, one node | same expression, elementwise over the whole output vector |
| `compute_output_delta` (cross-entropy/softmax): `a - y` | elementwise subtraction |
| `compute_hidden_delta`: Python `sum(node.delta * node.input_node_weights[own_index] for node in next_layer_nodes)`, one node at a time | `downstream = W_next.T @ delta_next` - the direct transposed analogue of the forward pass's own matrix multiply |
| sigmoid hidden delta: `downstream * a * (1-a)` | elementwise multiply |
| `ReLUNode.compute_hidden_delta`: `downstream if self.value() > 0.0 else 0.0`, per node | elementwise `where(a > 0.0, downstream, 0.0)` |

### gradient computation + weight update (`apply_gradient`)

| current code | vectorized replacement |
|---|---|
| `BackpropNode.apply_gradient`: nested Python loop, `weight - lr * delta * node.value()` per (node, input) pair | `grad_W = outer(delta, x)` (single example) - one call replaces the entire double loop; `W -= learning_rate * grad_W` |
| bias update: `bias - lr * delta`, per node | `b -= learning_rate * delta` (already vector-shaped) |
| mini-batch averaging (see `docs/structure.md`'s "mini-batch gradient descent" item) | `grad_W = (delta_batch.T @ x_batch) / batch_size` - one matrix multiply does the accumulate-then-average in one call instead of a Python accumulation loop across examples |
| `make_momentum_node_cls`: `lr*delta*x + momentum*prev`, per weight | elementwise arithmetic on arrays, plus a zero-initialized array for `_prev_weight_deltas` instead of a Python list |
| `make_l2_node_cls`: `lr*(delta*x + l2_lambda*weight)`, per weight | elementwise arithmetic |

### data loading (`mnist_data.py`) - arguably the single biggest real win

`load_mnist_dataset`/`load_mnist_records_at_indices` currently decode into
`tuple[float, ...]` per example - the exact "60000 x 784 = 47 million boxed Python float
objects" cost [research and analysis](research-and-analysis.md#parallelizing-mnist-training)
measured and fixed (via lazy per-worker decoding, not by avoiding the boxing itself).
Decoding raw `uint8` pixel bytes directly into one contiguous `float64` array
(`array = bytes.astype(float64) / 255.0`, computed once per loaded slice) would address the
root cause of that memory cost directly, rather than working around it the way
`load_mnist_records_at_indices`'s seek-and-decode-only-what's-needed approach currently does.

### initialization (`randomize`/`randomize_fan_in_aware`)

`random.uniform(-limit, limit)` in a nested Python loop, one weight at a time, becomes a
single call drawing an entire weight matrix at once from a seeded uniform distribution. Worth
flagging: this means leaving the stdlib `random` module's global seed state for a separate RNG
- a real behavior-affecting choice, not a transparent swap, since every seeded/pinned test in
this codebase currently seeds via `random.seed`.

### evaluation (`multiclass_evaluate.py`, `evaluate.py`)

`classify_state`'s `max(range(class_count), key=lambda i: probabilities[i])` and
`confusion_matrix`'s per-example loop both become one `argmax(axis=-1)` call over a whole
batch of activations, with the confusion matrix itself built from batched predicted/true label
arrays in one pass instead of a Python double loop.

## numerical-stability risks

- **`math.exp` raising `OverflowError` vs. a vectorized `exp` silently producing `inf`.**
  `sigmoid()`'s current `try/except OverflowError: return 0.0` (see its own docstring -
  deliberately not the piecewise reformulation, specifically to stay bit-identical to the
  un-clipped formula everywhere it doesn't overflow) relies on Python's `math.exp` raising.
  Most vectorized `exp` implementations never raise - they return `inf` for large positive
  input, and `1.0/(1.0+inf) == 0.0` falls out naturally via IEEE 754 semantics, which plausibly
  reproduces the same result - but that's a claim to check with the same kind of large
  random-input sweep the original guarded-sigmoid work used (200k random `z` values, checked
  bit-identical), not something to assume transfers.
- **Summation/accumulation order.** A hand-written reduction loop may not associate additions
  in the same order Python's own `sum()` does, which can produce a different (but equally
  valid) float64 rounding in the last bit or two - the same category of problem that got an
  earlier piecewise-sigmoid attempt reverted in this codebase (it changed rounding across the
  whole negative range and broke a pinned regression test, see
  [structure](structure.md#backprop)). Every new vectorized operation needs its own
  random-sweep parity check against the existing pure-Python reference before being trusted,
  not just "looks right."
- **dtype.** Must be explicit `float64` throughout, matching Python's own `float`, for the same
  pinned-test-fragility reason.

## decision: a hand-built Rust core (via PyO3/maturin), not C, not real NumPy

Two implementation languages were compared for a from-scratch array core covering exactly the
function inventory above (deliberately not general-purpose NumPy): C wrapped via `ctypes` or
the raw CPython C-API, and Rust wrapped via [PyO3](https://pyo3.rs)/
[maturin](https://www.maturin.rs). Rust came out ahead primarily on the Python-binding layer,
not the numerical code itself (the actual math - matmul, elementwise ops, reductions - is the
same algorithmic work in either language): PyO3's `#[pyclass]`/`#[pyfunction]`/`#[pymodule]`
macros generate the reference-counting and GIL-handling boilerplate automatically, eliminating
the single riskiest, hardest-to-debug part of a raw CPython C-API extension (manual
reference-counting bugs - dangling references, double-frees, refcount leaks), and `maturin`
absorbs the packaging/build step a C extension needs a separate `setup.py`/Makefile for. Real,
production-proven precedent exists for exactly this pattern (`polars`, `ruff`, `orjson`,
`cryptography`'s core are all Rust-behind-Python-bindings, not experiments).

PyO3/Rust is the toolchain this workplan targets; real NumPy remains the pragmatic default this
document is deliberately not recommending over it.

## workplan

### scope

Fixed `f64` dtype, up to 2D (matrix) + 1D (vector), row-major contiguous storage. Only the
operations actually exercised by this codebase per the inventory above: matmul, elementwise
`+ - * /` and comparisons, scoped broadcasting (matrix+row-vector, scalar - not general N-d
broadcasting, since this codebase never needs more than 2 dimensions), `exp`, `max`/`sum`/
`argmax` with axis reduction, `outer`, `where`, `clip`, a uniform RNG, raw-byte MNIST decode.

### 1. crate structure

A new top-level Rust crate (e.g. `rust/perceptron_array/`), kept separate from `perceptron/`
and standalone-buildable until proven correct, not dropped into the existing package tree
prematurely.

- `Cargo.toml`: `pyo3` (with the `extension-module` feature) as the only external crate -
  deliberately no `ndarray`, no `rand` crate, matching this repo's own "hand-build everything"
  posture and answering the same dependency-purity question consistently at this layer too
  (pulling in Rust crates for convenience would just move the same "adopt vs. hand-build"
  tension NumPy itself raises down one level, not resolve it).
- `src/array.rs` - the core type: a plain struct wrapping a flat `Vec<f64>` buffer plus a
  `shape: (usize, usize)` (or an enum distinguishing the 1D/2D cases actually used - no reason
  to build general N-dimensional machinery this codebase never needs).
- `src/ops.rs` - elementwise arithmetic + the scoped broadcasting rules.
- `src/linalg.rs` - matmul (naive triple loop first; a cache-friendlier loop order as a
  later, separately-measured refinement, not bundled into "get it correct" - matches this
  session's own "measure before assuming" posture, applied to performance instead of ML
  results). Not targeting BLAS-competitive performance - that's decades of industry SIMD/
  cache-tuning work, out of scope regardless of language; the realistic target is "meaningfully
  faster than pure Python."
- `src/ufuncs.rs` - `exp`, `max`/`sum`/`argmax` (with an axis parameter), `outer`, `where_`,
  `clip`.
- `src/random.rs` - a hand-rolled PRNG (a small, well-known public-domain algorithm - e.g.
  xorshift128+ or PCG32 - implemented directly, not pulled from the `rand` crate) plus a
  uniform-range array-fill function.
- `src/mnist.rs` - raw `u8` buffer -> `f64` array decode (the `/255.0` rescale), a direct port
  of what `mnist_data.py` does per-pixel today.
- `src/lib.rs` - the `#[pymodule]` entry point, exposing an array `#[pyclass]` (a
  project-specific name, avoiding collision with real NumPy's own `PyArray`) with
  `#[pymethods]` for the operations above, plus free `#[pyfunction]`s for the standalone ones
  (RNG, MNIST decode).

### 2. build integration into this repo's existing tooling

- `cli setup` gains a step installing `rustc`/`cargo` (matching its existing
  `apt install python3-tk` pattern) and `maturin` (via pip, into the same `.venv`).
- `cli setup` (or a new `cli build-rust`) runs `maturin develop` to build the extension into
  the active venv - the Rust-side analogue of `pip install -r requirements.txt`.
- `.gitignore` gains the Rust build output (`rust/*/target/`), matching the existing pattern
  for other regenerable artifacts (`trained_model.json`, `__pycache__`).
- `requirements.txt` stays Python-only; `Cargo.toml` is the Rust-side equivalent, not folded
  into it.

### 3. build order

Each stage independently testable before the next starts - no stage depends on an unvalidated
earlier one:

1. Array construction/shape/indexing only, no math yet - prove the Python<->Rust round-trip
   (construct from a Python list, read values back) works first.
2. Elementwise ops + broadcasting.
3. `exp`/`clip` together (needed together for the sigmoid-overflow question above).
4. matmul.
5. `max`/`sum`/`argmax` with axis reduction, `outer`, `where`.
6. RNG.
7. MNIST byte decode.

### 4. numerical parity validation

For every operation: a randomized-input test comparing the Rust result against the existing
pure-Python reference implementation (`BackpropNode.z()`, `sigmoid()`,
`SoftmaxOutputLayer.forward()`, etc.) across a large random-input sweep - the same discipline
as the 200k-random-z bit-identical sigmoid check already precedented in this codebase, extended
to every new operation rather than assumed to transfer. The two known risk areas (summation
order, sigmoid-overflow behavior) from the "numerical-stability risks" section above need this
check explicitly and early, not as an afterthought.

### 5. what stays explicitly out of scope for this workplan

- **Rewiring `BackpropNode`/`BackpropLayer` to actually use the new array type.** As flagged
  above, this is a separate, larger architectural change - it touches the shared base classes
  directly, unlike every additive sibling built so far in this codebase, and shouldn't start
  until the array core is independently proven correct. A follow-on workplan of its own, once
  this stage is done and trusted.
- **BLAS-competitive matmul performance.** Naive-but-correct first; any tuning is a
  separately-measured follow-up, not bundled into "does it work."
- **Any migration of existing pure-Python classes.** Nothing in `perceptron/model/` changes as
  part of this plan - the extension is built and validated as a fully standalone, parallel
  unit, consistent with how every technique in this codebase has been added: measured and
  proven in isolation before touching anything that already works.

## what this document is not

An analysis and a workplan, not an implementation, not a migration plan, and not a decision to
build. Whether to pursue this at all - vs. plain `numpy`, vs. leaving the codebase pure Python
- remains the explicitly flagged, undecided question in
[structure](structure.md#possible-next-steps): any array-library dependency, hand-built or
adopted, is a deliberate choice for whoever maintains this repo, not something to assume is
wanted just because it would be faster.
