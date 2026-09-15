# vectorized array-based model classes

[← back to vectorization](vectorization.md)

Part 1 of 3 in the [vectorization](vectorization.md) workplan split: what new classes would
look like if this codebase's forward/backward/gradient math were rewritten around whole-layer
arrays instead of individual node objects. Analysis and workplan only - not a decision to build
anything, and nothing in `perceptron/` changes as a result of this document. Deliberately scoped
to use real `numpy` as the concrete array backend for this stage - see "why numpy here, now" -
not a decision to adopt it as a permanent dependency; see
[the numpy interface subset](numpy-interface-subset.md) and
[the Rust implementation plan](rust-array-core.md) for how a hand-built replacement would take
its place later.

## why numpy here, now

[vectorization](vectorization.md)'s original analysis scoped a hand-built Rust array core
specifically to avoid adopting real `numpy` as a permanent dependency - a deliberate values
choice for this "hand-build everything" codebase, not a technical necessity. But *designing and
proving out* a whole new class of array-based model classes is a separable question from
*which array implementation eventually backs them* - and answering it against a hand-rolled
Rust core that doesn't exist yet would mean debugging two unproven things at once (the new
class architecture, and a brand-new array library) with no working reference to check either
against. Building these classes against real numpy first - already installed on this machine,
already proven correct and fast (see vectorization.md's own 150.7x-speedup benchmark) - lets
the class design and its numerical parity against the existing pure-Python reference be proven
independently, before [the Rust implementation](rust-array-core.md) has to reproduce anything.
Once that Rust core exists and passes its own parity checks (against numpy itself, transitively
validating against the same pure-Python reference these classes are checked against here), swapping
these classes' backend from `numpy` to it is the natural, explicitly out-of-scope follow-on this
document defers - see "what stays out of scope" below.

## the architectural point: a standalone network, not a BackpropNetworkBase sibling

Every sibling class built so far in this codebase - including
[convolutional layers](convolutional-layers.md)'s `ConvLayer`, which shares weights across many
spatial positions - still represents "one trainable unit" as one Python object with a `.value()`/
`.delta` pair, and reuses `BackpropNetworkBase`'s generic per-node/per-layer orchestration
(`_forward_outputs`, `_backward_hidden_layers`, the accumulate/apply/snapshot hooks - see
[mini-batch gradient descent](mini-batch-gradient-descent.md)'s and
[convolutional layers](convolutional-layers.md)'s own "architectural point" sections). That
machinery is built around `compute_hidden_delta(next_layer_nodes, own_index)` - one call per
node, indexing into a downstream node's own per-node `input_node_weights` list.

Array-based vectorization can't reuse that. The entire point is replacing "one Python object,
one method call, per node" with "one array, one matrix operation, for the whole layer" -
`vectorization.md`'s own "architectural point" section already says this plainly: *"this would
need to change what a node's state even is, structurally."* A layer's weights become one 2D
`ndarray` (`shape=(size, input_size)`), not `size` separate `list[float]`s; a layer's
activations become one 1D or batched-2D `ndarray`, not `size` separate node objects each
caching its own `._activation`. There is no `own_index` to enumerate and no per-node
`compute_hidden_delta` to call - the entire backward pass for a layer is one matrix
multiplication (`W_next.T @ delta_next`), computed once, not `size` times.

Given that, this workplan's `VectorizedMultiClassBackpropClassifierNetwork` is a **standalone
class**, not a `BackpropNetworkBase` subclass and not built from `ArrayLayer`/`BackpropLayer`
composition the way `ConvMultiClassBackpropClassifierNetwork` reused as much of the existing
generic machinery as it could. It shares only the *external* contract every sibling network in
this codebase already shares - `learn`, `learn_batch`, `classify_state`, `predict_probabilities`,
`snapshot`/`restore`, `save`/`load` - not any internal implementation. This is a genuinely
different shape of addition than anything built in this codebase so far, and is exactly why
`vectorization.md`'s own original doc scoped rewiring the existing classes as explicitly out of
scope for its own workplan - this document is that follow-on, still deliberately *not* touching
`perceptron/model/backprop_node.py`/`backprop_layer.py`/`backprop_network_base.py` themselves;
`VectorizedMultiClassBackpropClassifierNetwork` is purely additive, a new, fully independent
class.

## class design

### `ArrayLayer` - one layer's weights as one array, not `size` node objects

| method | array formula |
|---|---|
| `__init__(size, input_size)` | `self.W = np.zeros((size, input_size))`, `self.b = np.zeros(size)` - shape, not a list of node objects |
| `forward(x)` (single example) | `self.z = self.W @ x + self.b; self.a = sigmoid(self.z)` (or ReLU/softmax - a `_activation` override point, one per network variant, not per node) |
| `forward_batch(X)` (`X.shape == (batch_size, input_size)`) | `self.Z = X @ self.W.T + self.b; self.A = sigmoid(self.Z)` - the same formula, batched, via `numpy`'s own broadcasting of `+ self.b` across every row |
| `compute_output_delta(reference)` (quadratic) | `self.delta = (self.a - reference) * self.a * (1 - self.a)` - elementwise, whole vector at once |
| `compute_hidden_delta(next_layer)` | `downstream = next_layer.W.T @ next_layer.delta; self.delta = downstream * self.a * (1 - self.a)` - one matmul replaces `BackpropNode.compute_hidden_delta`'s per-node Python `sum()` entirely |
| `accumulate_gradient()` (single example, called once per example in a batch) | `self._grad_W += np.outer(self.delta, self.input_layer.a); self._grad_b += self.delta` - mirrors `BackpropNode.accumulate_gradient`'s own accumulate/apply split (see [mini-batch gradient descent](mini-batch-gradient-descent.md)), one array op instead of a double Python loop |
| `apply_accumulated_gradient(learning_rate, batch_size)` | `self.W -= learning_rate * self._grad_W / batch_size; self.b -= learning_rate * self._grad_b / batch_size`, then zero the accumulators - the exact same accumulate/apply *shape* as every other sibling in this codebase, just array-valued instead of per-node |

### `VectorizedMultiClassBackpropClassifierNetwork` - the standalone network

| method | design |
|---|---|
| `__init__(layer_sizes, dimension, class_count)` | builds a list of `ArrayLayer`s, matching `MultiClassBackpropClassifierNetwork`'s own shape (hidden layers, then a `class_count`-wide output layer) but with no `StateLayer`/`BackpropLayer` involved at all |
| `_forward(state)` | `x = np.array(state, dtype=np.float64)`, then each layer's `forward(x)` in sequence, `x = layer.a` between them - the batched analogue (`forward_batch`) is what `learn_batch` uses instead |
| `learn(learning_rate, state, category)` | forward, `compute_output_delta` against a one-hot target array (`target = np.zeros(class_count); target[category] = 1.0` - a plain zero-fill plus a single index write, not a general indexing operation), `compute_hidden_delta` for every earlier layer in reverse, `accumulate_gradient` + `apply_accumulated_gradient(batch_size=1)` per layer - the array-valued analogue of `BackpropClassifierNetwork.learn`'s own three-line body |
| `learn_batch(learning_rate, batch)` | stacks every example's state into one `X` matrix and every category into a one-hot `Y` matrix, runs `forward_batch`, computes every layer's delta batched (`compute_hidden_delta` becomes a `(batch_size, size)`-shaped array op, not a Python loop over examples), accumulates once per layer instead of once per (layer, example) pair - `grad_W = (delta_batch.T @ X_batch) / batch_size` replaces `mini-batch gradient descent`'s own per-example accumulation loop with one matrix multiply, per `vectorization.md`'s own original inventory |
| `randomize()` | fan-in-aware, `limit = 1/sqrt(previous_size)`, `layer.W = np.random.uniform(-limit, limit, size=(size, previous_size))` - one call per layer instead of a nested Python loop over every weight (see `vectorization.md`'s own "initialization" inventory entry for the RNG-state caveat this carries over unchanged) |
| `classify_state(state)` | `np.argmax(self.predict_probabilities(state))` - `np.argmax` replaces `max(range(class_count), key=...)` |
| `snapshot()`/`restore()` | `[(layer.W.copy(), layer.b.copy()) for layer in self.layers]` and the inverse - array-valued, but the same shape every other sibling's snapshot already has |
| `save()`/`load()` | its own JSON envelope (arrays serialized via `.tolist()`/`np.array(...)` on load) - can't reuse `save_model_json` any more than `ConvMultiClassBackpropClassifierNetwork` could (see that class's own identical finding) |

### MNIST data loading (`mnist_data.py`) - flagged by the original doc as the single biggest real win

`load_mnist_dataset`/`load_mnist_records_at_indices` currently decode into `tuple[float, ...]`
per example - the exact "60000 x 784 = 47 million boxed Python float objects" cost
[research and analysis](research-and-analysis.md#parallelizing-mnist-training) measured and
worked around (lazy per-worker decoding), not fixed at the root. A parallel
`load_mnist_dataset_as_array(path, limit=None) -> np.ndarray` (shape `(n, 784)`), decoding raw
`uint8` pixel bytes directly via `np.frombuffer(data, dtype=np.uint8).reshape(n, 785)[:, :-1]
.astype(np.float64) / 255.0`, addresses the root cause directly - a new, additive function
alongside the existing one, not a replacement (nothing currently calling
`load_mnist_dataset` needs to change).

## numerical parity validation

Every method above needs the same discipline `vectorization.md`'s own "numerical-stability
risks" section already lays out, checked here against the *existing pure-Python reference
classes* (`BackpropNode`/`MultiClassBackpropClassifierNetwork`), not assumed from the formulas
looking equivalent:

- **`sigmoid`'s overflow behavior.** `1/(1+math.exp(-z))`'s current `try/except OverflowError:
  return 0.0` vs. numpy's `exp` returning `inf` (and `1/(1+inf) == 0.0` falling out via IEEE 754)
  - needs the same large random-`z` sweep the original guarded-sigmoid work used, not an
  assumption.
- **Summation-order rounding.** `np.outer`/`@`'s internal reduction order isn't guaranteed to
  match Python's own `sum()` - the same category of float64 rounding-order risk that got an
  earlier piecewise-sigmoid attempt reverted in this codebase (see
  [structure](structure.md#backprop)).
- **A required regression gate, mirroring mini-batch's own batch-size-1 parity check**: for a
  fixed random seed, weights, and input, `VectorizedMultiClassBackpropClassifierNetwork.learn()`
  must land on the *same* (or provably float64-noise-close) weights as
  `MultiClassBackpropClassifierNetwork.learn()` - across a large randomized sweep, not a single
  hand-picked case, before any speed or accuracy claim is trusted.

## validation targets

The same two real, already-measured baselines every other workplan in this codebase's docs
validates against: UCI digits (8x8, `MultiClassBackpropClassifierNetwork`'s 99.5%/96.9%
train/test) first, cheap enough for fast iteration; real MNIST
(`docs/research-and-analysis.md`'s 92.75%/89.12%/96.01% figures) second, once small-scale parity
is proven. This workplan's own claim to check is *identical* accuracy trajectories to the
existing pure-Python classes at meaningfully lower wall-clock cost - not a new capability, a
faster path to the same numbers, honestly measured rather than assumed from the formulas' own
algebraic equivalence.

## workplan

### 1. `ArrayLayer` forward pass, single example

Construction, `forward`, parity-checked against `BackpropNode.z()`/`sigmoid()` across a random
sweep - no batching, no backward pass yet.

### 2. `ArrayLayer` forward pass, batched

`forward_batch`, parity-checked against running the single-example path once per row of a random
batch and stacking the results.

### 3. backward pass (single example, then batched)

`compute_output_delta`/`compute_hidden_delta`, parity-checked against
`BackpropNode.compute_output_delta`/`compute_hidden_delta` on matching random weights/deltas.

### 4. gradient accumulate/apply (single example, then batched)

`accumulate_gradient`/`apply_accumulated_gradient`, parity-checked against
`BackpropNode`'s own pair (see [mini-batch gradient descent](mini-batch-gradient-descent.md))
at `batch_size=1`, then against `train_backprop_network_mini_batch`'s own batched result at
`batch_size>1`.

### 5. `VectorizedMultiClassBackpropClassifierNetwork` end to end

`learn`/`learn_batch`/`randomize`/`classify_state`/`snapshot`/`restore`/`save`/`load`, assembled
from stages 1-4's already-parity-checked pieces. `load_mnist_dataset_as_array` alongside it, its
own parity check against `load_mnist_dataset`'s existing decode.

### 6. the actual validation this workplan exists to produce

Train on UCI digits, multiple seeds, confirming an *identical* accuracy trajectory (not just a
plausibly-similar one) to `MultiClassBackpropClassifierNetwork` at the same architecture, plus a
direct wall-clock comparison - then real MNIST, once small-scale parity is proven, checking
`vectorization.md`'s own extrapolated "~4 minutes to under 15 seconds per epoch" claim against
a real measurement.

### 7. what stays explicitly out of scope

- **Any change to `perceptron/model/backprop_node.py`/`backprop_layer.py`/
  `backprop_network_base.py`, or any existing sibling class.** Purely additive - see "the
  architectural point" above.
- **Swapping the array backend from `numpy` to the Rust core.** That's
  [the Rust implementation plan](rust-array-core.md)'s own eventual follow-on, gated on that
  core existing and passing its own parity checks first - not something this document does or
  assumes.
- **Convolutional or other sibling variants of the vectorized classes** (a vectorized
  `ConvLayer`, momentum, L2, etc.). This workplan proves out the base
  one-vs-rest/quadratic-loss case only; every other variant is its own, separate follow-on once
  this one is trusted.

## what this document is not

An analysis and a workplan, not an implementation, not a decision to adopt `numpy` as a
dependency. Whether to pursue this at all - and whether `numpy` ever becomes more than this
workplan's own temporary prototyping vehicle - remains the explicitly flagged, undecided
question [structure](structure.md#possible-next-steps) already raises for vectorization as a
whole.
