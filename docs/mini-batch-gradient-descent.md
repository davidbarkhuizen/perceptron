# mini-batch gradient descent

[← back to README](../README.md)

## status: infrastructure built, the retest it was built for is not

Stages 1-5 of the workplan below are built (`accumulate_gradient`/`apply_accumulated_gradient`
on every trainable node, `learn_batch` on both classifier networks, `train.
train_backprop_network_mini_batch`, and the numerical parity tests each stage required) - see
the three PRs that implemented them (#139, #140, #141). The rest of this document is kept as
written before that work started, since it's still the accurate record of the reasoning and
the function-level detail behind what got built - **only this status section reflects
present-tense reality**; everything below still reads as a forward-looking plan for stages
that are, as of PR #141, actually finished. **Stage 6 - the momentum re-test this entire
workplan exists to unblock - has not been run.** Building the mini-batch machinery answers the
architectural "how would this even work" question; it doesn't itself tell us whether
mini-batching helps momentum, which is the actual open question. See
`docs/structure.md`'s "possible next steps" for the current status of that still-open retest.

## original framing (superseded by "status" above for what's actually built)

Analysis and workplan only - not a decision to build anything here, and nothing in
`perceptron/` changes as a result of this document. `docs/structure.md`'s "possible next
steps" lists mini-batch gradient descent as still open, directly motivated by a specific
open question left by an existing measured result (momentum - see below); this document
works that motivation through to a concrete implementation plan, in the same spirit as
[vectorization](vectorization.md)'s workplan for a hand-built array core.

## why this is next: the open question it answers

`MomentumBackpropClassifierNetwork` exists and is tested, but its own measurement
(`research-and-analysis.md`'s "momentum" entry) found momentum's canonical coefficient
(α=0.9) robustly *hurt* across a learning-rate sweep, and a finer sweep (0.3-0.7, n=15 seeds)
landed statistically indistinguishable from plain SGD - a flat null, not a win. The leading
candidate explanation recorded there: this codebase's per-example *online* SGD produces
individual gradients far noisier than the batch/mini-batch gradients momentum's own literature
is validated against, so accumulating velocity across noisy individual steps amplifies noise
instead of smoothing signal. `MomentumBackpropClassifierNetwork` was deliberately kept, not
discarded, specifically to retest under mini-batch gradients once they exist (see
`momentum_backprop_classifier_network.py`'s own docstring, `docs/structure.md:354`). This
document exists to make that retest actually possible - it isn't itself a performance
workplan (see "expected effect" below for why pure-Python mini-batching isn't primarily about
speed).

## the architectural point that matters more than any single function

Every trainable node's `apply_gradient` (`BackpropNode.apply_gradient`,
`perceptron/model/backprop_node.py`) does two things in one call, back to back: compute this
example's gradient from `self.delta` and each input node's *current* `value()`, and
immediately write the updated weight. There is no seam between "compute a gradient" and
"apply it" - `apply_gradient` **is** the update. Mini-batching needs exactly that seam: run
forward+backward for each example in a batch, **accumulate** each one's gradient without
touching the weights, then apply the batch-averaged gradient once. Every existing sibling that
touches gradient application doesn't call the base method and adjust its result - each fully
replaces `apply_gradient` with its own formula:

- `MomentumBackpropNode.apply_gradient` (`momentum_layer.py`) - folds in `momentum * prev` and
  updates `_prev_weight_deltas`.
- `L2RegularizedBackpropNode.apply_gradient` (`l2_regularization_layer.py`) - adds
  `l2_lambda * weight` to the gradient.

Both of these need the identical accumulate/apply split, not just the plain node - three call
sites, not one. This is a different shape of change than every additive sibling built so far
in this codebase (ReLU, cross-entropy, softmax all override a per-node method's *body* without
changing when or how often it's called); like the vectorization workplan's own architectural
point, this one changes what the training loop's *unit of work* is, not just what formula runs
inside it.

## function-level inventory

Grounded in the actual current implementation (`perceptron/model/backprop_node.py`,
`backprop_layer.py`, `backprop_network_base.py`, `backprop_classifier_network.py`,
`multiclass_backprop_classifier_network.py`, `momentum_layer.py`, `l2_regularization_layer.py`,
`perceptron/train.py`).

### node-level gradient application

| current code | mini-batch replacement |
|---|---|
| `BackpropNode.apply_gradient(learning_rate)`: reads `self.delta`/`node.value()` and writes the weight in one step, once per example | split into `accumulate_gradient()` (adds `self.delta * node.value()` per input, and `self.delta` for bias, into new `_weight_gradient_accum`/`_bias_gradient_accum` fields - no weight write) and `apply_accumulated_gradient(learning_rate, batch_size)` (`weight -= learning_rate * accum / batch_size`, then zeroes the accumulator) |
| `MomentumBackpropNode.apply_gradient`: `delta_w = lr*delta*x + momentum*prev`, applied per example | same split; the averaged data-gradient (`accum / batch_size`) plugs into the existing formula in place of the single-example term, so momentum's velocity update happens once per batch, not once per example - this is the actual mechanism change the "why this is next" section above is testing |
| `L2RegularizedBackpropNode.apply_gradient`: `weight -= lr*(delta*x + l2_lambda*weight)` per example | same split, with `l2_lambda * weight` added once at apply time (using the weight's value at that point), not accumulated per example inside the batch - see "numerical and behavioral risks" for why per-example accumulation of this term would be wrong |

### layer/network-level orchestration

| current code | mini-batch replacement |
|---|---|
| `BackpropNetworkBase._apply_gradients(learning_rate)`: one pass over every trainable node, calling `apply_gradient` | becomes `_accumulate_gradients()` (one pass calling `accumulate_gradient()`, run once per example inside a batch) plus `_apply_accumulated_gradients(learning_rate, batch_size)` (one pass calling `apply_accumulated_gradient`, run once per batch) |
| `BackpropClassifierNetwork.learn`/`MultiClassBackpropClassifierNetwork.learn`: `forward -> backward -> _apply_gradients`, once per example | unchanged, kept as the batch-size-1 fast path (see numerical risk below on why this needs proving, not assuming) - each class also gains `learn_batch(learning_rate, batch)`: `for example in batch: forward, backward, accumulate` then one `_apply_accumulated_gradients` call |
| `perceptron/train.py`'s `train_linear_classifier_network`: `for datum in training_data: student.learn(...)`, one example at a time | **not** retrofitted in place - this function is shared with `LinearClassifierNetwork`/`AssociationNode`, whose discrete minimum-disturbance update rule has no gradient or batch concept at all (see `demo_xor_linear_classifier_ceiling.py`); mini-batching needs its own training-loop function for gradient-based students, not a branch bolted into the existing generic one |
| (new) batch construction | a chunking helper that splits `training_data` into `batch_size`-sized groups per epoch - open design questions below (last partial batch, per-epoch reshuffle) |

## numerical and behavioral risks

- **Batch-size-1 parity is a required regression gate, not an assumption.** 225 existing tests
  are pinned against today's exact per-example update rule. The accumulate/apply split must
  reproduce `learn()`'s current bit-for-bit output at `batch_size=1` before anything else is
  trusted - a random-weight/random-input sweep comparing old `apply_gradient` against new
  `accumulate_gradient` + `apply_accumulated_gradient(batch_size=1)`, for all three node
  variants (plain, momentum, L2), the same discipline `vectorization.md`'s own numerical-risk
  section applies to its Rust operations.
- **Accumulation-order rounding.** Summing gradients across a batch in a Python loop may not
  associate identically to how a single-example update would (trivial at `batch_size=1`, a real
  question at `batch_size>1`) - the same category of float64 rounding-order issue that got an
  earlier piecewise-sigmoid change reverted in this codebase (see
  `structure.md#backprop`). Not expected to matter beyond the last bit or two, but a claim to
  check, not assume.
- **L2's penalty term must not be accumulated per example.** The weight itself doesn't change
  during a batch's forward/backward passes (only applied once, at the end), so `l2_lambda *
  weight` is the same value at every example in the batch. Accumulating it once per example and
  averaging (`sum(l2_lambda*weight for _ in batch) / batch_size`) happens to reduce back to the
  same single term mathematically - but naively accumulating it *unaveraged*, or getting the
  order of operations wrong relative to the data-gradient average, would silently over- or
  under-penalize by a factor of `batch_size`. Worth its own explicit unit test, not just
  inference from the algebra.
- **Averaging vs. summing changes the effective step size.** `grad / batch_size` (averaging,
  matching `vectorization.md`'s own `grad_W = (delta_batch.T @ x_batch) / batch_size`) keeps
  `learning_rate`'s meaning comparable to today's tuned per-example values as `batch_size`
  varies; summing instead would make the effective step scale with `batch_size` and require
  re-tuning `learning_rate` per batch size to compare fairly. This document picks averaging
  for consistency with the vectorization workplan's own formula, but it's a real design
  decision, not the only valid one.
- **Momentum's own premise is what's being tested here, not assumed to now work.** The whole
  point of this workplan is to re-run momentum's measurement under lower-noise batch gradients
  - the outcome (does momentum help now?) is genuinely open, not a predetermined result to
  build toward.

## expected effect: this is not a speed workplan

Unlike `vectorization.md`, pure-Python mini-batching does the same total number of arithmetic
operations as today's per-example loop - forward/backward still runs once per example, one
Python object at a time; only the *timing* of the weight write changes (once per batch instead
of once per example). No meaningful wall-clock speedup should be expected from this workplan
on its own, and it may even be marginally slower (extra accumulator bookkeeping per example).
The actual goal is unblocking the momentum retest above, plus laying groundwork that's a
prerequisite for (but independent of) `vectorization.md`'s own batched design: that document's
`grad_W = (delta_batch.T @ x_batch) / batch_size` only becomes a meaningful replacement once
the training loop is already batch-shaped at the Python level - vectorizing an already-batched
loop is a separate, later step, not something this workplan does itself.

## decision

Unlike `vectorization.md` (still a pure analysis-and-workplan document, nothing built), this
workplan's infrastructure stages (1-5) *were* built - see "status" at the top. What's still
undecided is the same as it always was: whether momentum should ever be turned on by default,
which depends entirely on stage 6's retest result, not yet run. Building the machinery to run
that retest was judged worth doing on its own terms (the mini-batch capability itself, not
just the retest); whether the retest's eventual result changes anything about momentum's
default status remains for whoever maintains this repo to decide once it exists.

## workplan

### scope

Gradient-based students only (`BackpropClassifierNetwork`, `MultiClassBackpropClassifierNetwork`,
and their momentum/L2 siblings) - `LinearClassifierNetwork`'s discrete update rule is explicitly
untouched. Additive: `learn()`'s existing per-example path stays available and is the
`batch_size=1` case in behavior, proven by the parity check above, not replaced.

### 1. gradient accumulator fields

`BackpropNode` gains `_weight_gradient_accum: list[float]` and `_bias_gradient_accum: float`,
zero-initialized alongside `_prev_weight_deltas`-style state, reset to zero after each
`apply_accumulated_gradient` call - mirroring how `MomentumBackpropNode` already carries
`_prev_weight_deltas` as per-node mutable state.

### 2. split `apply_gradient` into accumulate/apply, at all three call sites

`BackpropNode` (`backprop_node.py`), `MomentumBackpropNode`
(`make_momentum_node_cls`, `momentum_layer.py`), `L2RegularizedBackpropNode`
(`make_l2_node_cls`, `l2_regularization_layer.py`). Each gets `accumulate_gradient()` (no
weight write) and `apply_accumulated_gradient(learning_rate, batch_size)` (averages, applies,
resets), replacing what `apply_gradient` currently does in one step. `apply_gradient` itself
can stay as a thin `accumulate_gradient(); apply_accumulated_gradient(learning_rate, 1)`
wrapper so `learn()`'s call sites need no change.

### 3. layer/network-level orchestration

`BackpropNetworkBase` gains `_accumulate_gradients()` and
`_apply_accumulated_gradients(learning_rate, batch_size)`, mirroring the existing
`_apply_gradients`/`_backward_hidden_layers` pattern. `BackpropClassifierNetwork` and
`MultiClassBackpropClassifierNetwork` each gain `learn_batch(learning_rate, batch)`: loop
`forward` + `_backward` + `_accumulate_gradients` per example, then one
`_apply_accumulated_gradients` call - the direct batch-shaped analogue of today's `learn()`.

### 4. batch construction

A new chunking helper (not a retrofit of `train_linear_classifier_network`, per the inventory
above) splitting `training_data` into `batch_size`-sized groups per epoch. Two open questions
to settle during implementation, not before: whether a final undersized batch is kept (averaged
over its own smaller size) or dropped, and whether `training_data` is reshuffled each epoch
(the existing per-example loop doesn't reshuffle; standard mini-batch SGD practice does, to
avoid the same batch composition recurring every epoch).

### 5. numerical parity validation

The batch-size-1 regression gate from "numerical and behavioral risks" above, run before
anything else: a random-weight/random-input sweep comparing old `apply_gradient` against the
new split, for all three node variants, required to pass before trusting any `batch_size>1`
result. The L2-accumulation-order unit test from the same section is part of this stage, not a
follow-up.

### 6. the actual retest this workplan exists to unblock

Once stages 1-5 are trusted: re-run `research-and-analysis.md`'s momentum investigation - same
statistical discipline (many seeds, mean + stdev, the n=15 sweep that caught the earlier n=5
mirage) - at a handful of batch sizes (e.g. 8, 32, 128) crossed with the same momentum
coefficients already measured (0.0, 0.3-0.7, 0.9), to test directly whether lower-noise batch
gradients let momentum's literature-cited benefit show up here, rather than assuming it will.

### 7. what stays explicitly out of scope for this workplan

- **Vectorized/batched matmul.** `grad_W = (delta_batch.T @ x_batch) / batch_size` and any
  NumPy- or Rust-backed batch computation belong to `vectorization.md`'s own separate workplan
  - this one is pure-Python, algorithmic-only, and produces the batch-shaped loop that
  workplan would eventually replace the internals of.
- **Changes to `ensemble_train.py`'s parallelization.** That pipeline is already parallel by
  class (one process per binary sub-classifier); mini-batching is an orthogonal, within-worker
  change to how each sub-classifier's own training loop is shaped, not a change to how workers
  are split.
- **Any change to `LinearClassifierNetwork`/`AssociationNode`.** Their minimum-disturbance
  update rule has no gradient to batch.
- **Re-tuning `learning_rate` for any specific batch size beyond what stage 6 needs.** A
  broader learning-rate-vs-batch-size sweep, if warranted at all, is a follow-on to whatever
  stage 6 finds, not part of this workplan.

## what this document is not

As of PR #141, no longer purely an analysis and a workplan for stages 1-5 - see "status" at the
top. Still not a migration plan (nothing in `BackpropNode`/`BackpropLayer`'s existing
per-example `learn()` path was removed or changed - `learn_batch` sits alongside it) and still
not a decision that momentum should be turned on: that remains the explicitly flagged,
undecided question in [structure](structure.md#possible-next-steps), gated on stage 6's retest,
which is what the built infrastructure exists to make possible - not yet what it has answered.
