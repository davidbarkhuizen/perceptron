# convolutional layers, from scratch

[← back to README](../README.md)

Analysis and workplan only - not a decision to build anything here, and nothing in
`perceptron/` changes as a result of this document. `docs/structure.md`'s "possible next
steps" flags convolutional layers as "a substantial but well-motivated next architecture step
... needing new node/layer abstractions for 2D receptive fields and weight sharing" - this
document works that through to a concrete, function-level plan, in the same spirit as
[vectorization](vectorization.md) and [mini-batch gradient descent](mini-batch-gradient-descent.md).

## why this is next

This codebase's only two real datasets - the bundled UCI digits (8x8, `digits_data.py`) and
real MNIST (28x28, `mnist_data.py`) - are both images, currently classified with dense layers
alone: every node in a `BackpropLayer` connects to *every* node in `input_layer`
(`BackpropLayer.__init__`: `input_nodes=self.input_layer.nodes`, the full list, for every
node), with fully independent weights per node. That throws away the one structural fact these
two datasets actually have that generic tabular data doesn't: pixels have 2D spatial
relationships, and the same local pattern (an edge, a stroke) can appear anywhere in the image.
A convolutional layer encodes that directly - local receptive fields instead of full
connectivity, and one shared, position-independent weight kernel per feature instead of one
independent weight set per node - matching this repo's existing pattern of progressively more
capable model classes (`LinearClassifierNetwork` -> `BackpropClassifierNetwork` -> multi-class
siblings) with a genuinely new structural capability, not just a new per-node formula.

## the architectural point that matters more than any single function

Every prior sibling layer in this codebase - `ReLULayer`, `SoftmaxOutputLayer`,
`CrossEntropyOutputLayer`, `MomentumLayer`/`L2Layer` from the mini-batch work - changes only a
per-node *formula* (a different `forward()`/`compute_output_delta()`/`apply_accumulated_gradient()`
body), while leaving two things completely fixed: every node in a layer connects to the *same*
full `input_layer.nodes` list, and every node owns its *own*, independent
`input_node_weights`. Convolution breaks both:

- **Local receptive fields.** A convolutional unit at output position `(row, col)` connects
  only to a `kernel_size x kernel_size` window of the input, not the whole layer - `input_nodes`
  becomes a small, position-dependent slice, not `input_layer.nodes` wholesale.
- **Weight sharing.** Every spatial position within one output channel uses the *identical*
  kernel weights and bias - not independently-initialized-and-trained values that happen to
  start the same, but the literal same trainable parameters, updated once per step from
  gradient contributions summed across every position that used them.

Weight sharing is the deeper break. `BackpropNode`/`WeightedInputNode` conflate "one spatial
computation" with "one independently-owned weight set" - `update_input_weights` rebinds
`self.input_node_weights` to a new list on that one instance, which can't express "this
position's weights are literally the same object every other position in this channel reads
and updates." The clean fix is to stop conflating the two: split "one spatial position's
forward/backward math" (a `ConvUnit`) from "one channel's shared, trainable kernel" (a
`ConvKernel`), where many `ConvUnit`s reference one `ConvKernel` and only the kernel accumulates
and applies gradient.

That reintroduces, in a new place, exactly the accumulate/apply seam
[mini-batch gradient descent](mini-batch-gradient-descent.md) just built - a kernel's gradient
is the *sum* of `delta * input_value` across every spatial position that used it this step,
computed by calling `accumulate_gradient()` once per position and `apply_accumulated_gradient()`
once per kernel, which is structurally identical to mini-batch's "accumulate once per example,
apply once per batch." But `BackpropNetworkBase`'s network-level orchestration
(`_accumulate_gradients`, `_apply_accumulated_gradients`, `_apply_gradients`, `snapshot`,
`restore` - see `backprop_network_base.py`) all iterate `for node in layer.nodes` directly, one
apply call per node:

```python
def _apply_accumulated_gradients(self, learning_rate: float, batch_size: int) -> None:
    for layer in self.trainable_layers:
        for node in layer.nodes:
            node.apply_accumulated_gradient(learning_rate, batch_size)
```

For a `ConvLayer`, `layer.nodes` is every spatial position (many `ConvUnit`s per `ConvKernel`) -
calling `apply_accumulated_gradient` once per *position* would apply (or, after the first
position empties the accumulator, silently no-op) the same kernel's update up to
`out_height * out_width` times per step, relying on iteration order rather than being correct
by construction. **These five methods need to delegate through a per-*layer* hook instead of
reaching into `layer.nodes` directly** - `BackpropLayer` gains default implementations that
are the exact loop bodies these methods already have today (a pure extraction, no behavior
change, verified the same way mini-batch's own split was: every existing pinned test must pass
unchanged), and `ConvLayer` overrides them to iterate `ConvUnit`s for accumulate (correct
as-is - summing every position's contribution into its kernel *is* the desired behavior) but
iterate `ConvKernel`s (once each) for apply/snapshot/restore. This is the one real
`BackpropNetworkBase`/`BackpropLayer` change this workplan needs - every previous sibling
needed zero.

## scoping v1: one conv layer, directly after the input

Stacking multiple convolutional layers needs backprop-through-convolution ("full convolution
with a flipped kernel," computing a gradient with respect to the *input* of a conv layer, not
just its weights) - real, standard CNN math, but genuinely new to this codebase, with no
existing analogue anywhere in it. A single conv layer placed directly after the (non-trainable)
input `StateLayer` needs none of that: nothing before it is ever trained, so there is nothing to
backpropagate *into*. `_backward_hidden_layers`'s existing formula -
`compute_hidden_delta(next_layer_nodes, own_index)`, called on every node in a layer from the
*next* layer's deltas - already works unchanged for a conv layer's own delta, computed from a
downstream *dense* layer's existing hidden-delta formula, because a dense layer downstream of
a (flattened) conv layer is just an ordinary fully-connected layer as far as that formula is
concerned - `own_index` still aligns correctly, since the flattened conv output is exactly what
the dense layer's `input_nodes` is. **v1 scope is deliberately narrowed to this specific,
much cheaper case**: exactly one `ConvLayer`, single input channel, `'valid'` padding
(no synthetic zero-padding nodes), a configurable stride, followed by ordinary dense hidden and
output layers, built on the real UCI digits (8x8) and real MNIST (28x28) datasets already in
this codebase. Stacked conv layers, multi-channel input, `'same'` padding, and pooling are all
explicitly out of scope for v1 - see "what stays out of scope" below.

## function-level inventory

Grounded in the actual current implementation (`perceptron/model/base_node.py`,
`backprop_layer.py`, `backprop_network_base.py`, `backprop_node.py`, `relu_layer.py`,
`multiclass_backprop_classifier_network.py`, `model_io.py`, `digits_data.py`, `mnist_data.py`).

### new: the shared kernel

| concept | design |
|---|---|
| `ConvKernel` (new class, one per output channel) | owns `weights: list[float]` (flat, length `kernel_size**2`) and `bias: float`, plus its own `_weight_gradient_accum`/`_bias_gradient_accum` and `accumulate_gradient(delta, receptive_field_values)`/`apply_accumulated_gradient(learning_rate, batch_size)` - the same shape as `BackpropNode`'s own pair (see `backprop_node.py`), just invoked once per contributing spatial position instead of once per training example, and applied once per kernel instead of once per node |
| kernel initialization | fan-in-aware, directly analogous to `randomize_fan_in_aware` (`backprop_network_base.py`): `limit = 1/sqrt(kernel_size**2 * in_channels)` - fan-in for a kernel is exactly its own receptive field size, not the whole previous layer |

### new: the spatial unit

| concept | design |
|---|---|
| `ConvUnit` (new class, one per output spatial position per channel) | **not** a `BackpropNode` subclass - composition, not inheritance, since `BackpropNode`'s `input_node_weights` is an owned, rebindable instance attribute, which fights a genuinely shared, cross-instance weight list rather than accommodating it. Implements the same duck-typed surface other nodes provide (`value()`, `forward()`, `delta`, `compute_hidden_delta()`) plus `accumulate_gradient()` (delegates into its `ConvKernel`) and a no-op `apply_accumulated_gradient()` (the kernel, not the unit, owns that step) |
| `forward()` | `z = sum(input_nodes[i].value() * kernel.weights[i] for i in range(kernel_size**2)) + kernel.bias`, then ReLU (`max(0.0, z)`) - the standard modern default for conv hidden layers, and this codebase already has the exact cached-activation-implies-derivative trick to reuse (`ReLUNode.compute_hidden_delta`, `relu_layer.py`: `downstream if self.value() > 0.0 else 0.0`) |
| receptive-field wiring | `ConvLayer.__init__` computes, for each output `(row, col)`, the flat input indices `[(row*stride + kr) * input_width + (col*stride + kc) for kr in range(kernel_size) for kc in range(kernel_size)]` into `input_layer.nodes` - row-major flat indexing, the same layout `mnist_data.load_mnist_dataset`/`digits_data.load_digits_dataset` already decode pixels into (an assumption worth stating explicitly, not silently relied on) |

### new: the layer

| concept | design |
|---|---|
| `ConvLayer` (new class - not a `BackpropLayer` subclass, for the same composition-over-inheritance reason as `ConvUnit`) | holds `channel_count` `ConvKernel`s and `out_height * out_width * channel_count` `ConvUnit`s as its `.nodes` (channel-major order, documented); `forward()` calls every unit's `forward()` |
| `accumulate_gradients()` | loop every `ConvUnit`, call `accumulate_gradient()` - correct as-is, since this *is* the desired "sum every position's contribution into its kernel" behavior |
| `apply_accumulated_gradients(learning_rate, batch_size)` | loop every `ConvKernel` (once each, not once per unit) |
| `snapshot()`/`restore()` | once per kernel (`weights`, `bias`), not once per unit |

### changed: `BackpropNetworkBase`/`BackpropLayer` (the one real shared-code change)

| current code | change |
|---|---|
| `_accumulate_gradients`, `_apply_accumulated_gradients`, `_apply_gradients`, `snapshot`, `restore` (`backprop_network_base.py`): each does `for layer in self.trainable_layers: for node in layer.nodes: node.<method>(...)` | each becomes `for layer in self.trainable_layers: layer.<method>(...)` - a per-layer hook instead of reaching into `.nodes` directly |
| `BackpropLayer` (`backprop_layer.py`) | gains default `accumulate_gradients()`/`apply_accumulated_gradients()`/`apply_gradients()`/`snapshot_state()`/`restore_state()` methods whose bodies are exactly today's extracted per-node loops - a pure refactor, zero behavior change for every existing layer type (dense, momentum, L2, ReLU, softmax) |
| `ConvLayer` | overrides the same five methods with the per-kernel semantics above |

### new: the network class

| concept | design |
|---|---|
| `ConvMultiClassBackpropClassifierNetwork` (new class, sibling of `MultiClassBackpropClassifierNetwork`, not a retrofit) | its own `__init__` (conv hyperparameters - `channel_count`, `kernel_size`, `stride`, `input_height`, `input_width` - plus `dense_layer_sizes: list[int]` for the layers after flattening, plus `class_count`), **not** calling `super().__init__()` - builds `self.input_layer`/`self.hidden_layers` (`[conv_layer] + dense_layers`)/`self.output_layer`/`self.trainable_layers` directly, matching the shape `BackpropNetworkBase`'s inherited methods (`_forward_outputs`, `_backward_hidden_layers`, the five gradient/persistence methods above) already expect - no change to *how* those methods work, only to what builds the attributes they read |
| `learn`/`learn_batch`/`_backward`/`classify_state`/`predict_probabilities`/`randomized` | inherited unchanged from `MultiClassBackpropClassifierNetwork` - none of them reach into layer internals directly, they all go through the methods this workplan already makes layer-polymorphic |
| `randomize()` | fan-in-aware for the conv kernels (see above) and for every dense layer, via the existing `randomize_fan_in_aware` for the dense portion |
| `save`/`load` | **cannot** reuse `save_model_json`'s envelope as-is - it hardcodes `layer_sizes: list[int]` (`model_io.py`), which has no way to express conv hyperparameters; needs its own envelope shape. A real, if peripheral, inventory item - not solved in depth here |

## numerical and behavioral risks

- **The five-method refactor is a required regression gate, exactly like mini-batch's own
  batch-size-1 parity check.** Pushing `_accumulate_gradients`/`_apply_accumulated_gradients`/
  `_apply_gradients`/`snapshot`/`restore`'s loop bodies down into `BackpropLayer` must not
  change behavior for any *existing* layer type - every one of the pinned tests this codebase
  already has (225 before mini-batch, more since) must pass unchanged before any `ConvLayer`
  code is trusted.
- **Gradient correctness needs numerical gradient checking, not just a hand-derived example.**
  A hand-computed tiny case (e.g. a 3x3 kernel over a 4x4 input) is a good first sanity check,
  the same way existing tests hand-derive expected values - but the standard, more rigorous tool
  for a genuinely new backward-pass formula is numerical gradient checking: perturb one kernel
  weight by a small `epsilon`, measure the resulting change in loss, and compare to
  `accumulate_gradient`'s analytic value. `numpy`'s own `scipy.signal.correlate2d`/manual
  convolution (or a hand-rolled equivalent), used purely as an independent offline oracle to
  cross-check the forward pass - not adopted as a dependency, the same "used purely as a proxy"
  framing [the Rust implementation plan](rust-array-core.md)'s own benchmark already
  establishes for this codebase - is a useful second check, but gradient checking is the one
  that actually validates the *backward* pass.
- **Row-major flat-index assumption.** `ConvLayer`'s receptive-field wiring assumes
  `input_layer.nodes[row * width + col]` matches standard raster order - true for both
  `mnist_data.py` (PNG decode, row by row) and `digits_data.py` (UCI's own published format),
  but worth a dedicated test against a small hand-constructed image with a distinctive pattern
  (e.g. a single hot pixel at a known `(row, col)`), not just assumed from the datasets' own
  conventions.
- **Whether conv actually helps is an open empirical question, not assumed.** `UCI digits at
  8x8 with `MultiClassBackpropClassifierNetwork`'s existing dense baseline already reaches
  99.5%/96.9% train/test (`randomize_fan_in_aware`'s own docstring) - a small, easy task that
  may already be close to saturated for a dense network, leaving little room for conv to show a
  clear win at this scale. This workplan's validation should measure directly, honestly
  reporting a null result the way momentum's and L2's own investigations did if that's what
  happens, not assume convolution wins because the literature says it generally does.

## expected effect and validation targets

Both of this codebase's real image datasets already have a measured dense-network baseline to
compare against directly:

- **UCI digits (8x8, 1797 examples)** - `MultiClassBackpropClassifierNetwork` with
  fan-in-aware init: 99.5% training / 96.9% test accuracy (`randomize_fan_in_aware`'s own
  docstring). Small and fast enough for the same kind of quick, iterative validation
  [the Rust implementation plan](rust-array-core.md)'s own throwaway benchmark and the
  mini-batch retest's proxy both used - the natural first target, not real MNIST.
- **Real MNIST (28x28, 60000/10000 train/test)** - `MultiClassBackpropClassifierNetwork`
  (92.75%), `SoftmaxMultiClassBackpropClassifierNetwork` (89.12%), and the
  `EnsembleBackpropClassifierNetwork` (96.01%) all have measured real-scale baselines
  (`docs/research-and-analysis.md`) - the eventual larger validation, once the UCI-digits scale
  proves the implementation correct, not the first place to test it (training-time cost alone
  argues for small-scale-first, the same reasoning `docs/mini-batch-gradient-descent.md`'s own
  proxy-then-real-scale pattern already follows).

Unlike `vectorization.md`, this isn't a performance workplan either - a from-scratch conv
layer's per-position Python loop is no faster than a dense layer's per-node loop at comparable
parameter counts (arguably slower for a deep stack, before any vectorization). The value case
is representational: fewer trainable parameters per output feature than a dense layer over the
same input (one shared `kernel_size**2`-weight kernel per channel, not `input_size` weights per
output node) and, per CNN theory, better generalization on image-shaped data from translation
invariance - both empirical claims to measure against the baselines above, not assume.

## decision: not made here

Same posture as `vectorization.md` and `mini-batch-gradient-descent.md` before it was built:
this document scopes one specific way convolutional layers could be added, in enough
function-level detail to implement, without recommending it be built. `docs/structure.md`'s
"possible next steps" already flags this as the most substantial of the remaining open items -
whether it's worth the real architectural work above (a genuine, if narrowly-scoped,
`BackpropNetworkBase`/`BackpropLayer` change, unlike any prior sibling) is for whoever
maintains this repo to decide.

## workplan

### scope

Exactly one `ConvLayer`, directly after the (non-trainable) input `StateLayer`, single input
channel, `'valid'` padding, configurable `stride`, ReLU activation, followed by ordinary dense
hidden/output layers - see "scoping v1" above for why this specific narrowing avoids needing
backprop-through-convolution entirely. `UCI` digits first, real MNIST second.

### 1. the layer-level hook refactor (do this first, independent of everything else)

Extract `BackpropNetworkBase`'s `_accumulate_gradients`/`_apply_accumulated_gradients`/
`_apply_gradients`/`snapshot`/`restore` loop bodies into new `BackpropLayer` methods
(`accumulate_gradients`/`apply_accumulated_gradients`/`apply_gradients`/`snapshot_state`/
`restore_state`), with the network-level methods becoming thin per-layer dispatchers. Zero
behavior change for any existing layer type - the full existing test suite (pinned tests
included) must pass unchanged before moving to stage 2. This stage has no dependency on
anything conv-specific and is valuable groundwork on its own, the same way each mini-batch
stage was independently testable before the next started.

### 2. `ConvKernel`

The shared per-channel weight/bias holder plus its own `accumulate_gradient`/
`apply_accumulated_gradient` pair (mirroring `BackpropNode`'s own shape from the mini-batch
work) and fan-in-aware initialization. Testable in complete isolation - no layer, no network,
just weight/gradient arithmetic against hand-computed values.

### 3. `ConvUnit`

Forward pass (receptive-field dot product against its kernel, ReLU) and the two backward-facing
methods (`compute_hidden_delta`, matching `ReLUNode`'s existing formula; `accumulate_gradient`,
delegating into its kernel). Testable against a single hand-constructed small input/kernel pair
with an independently-computed expected activation and, once a downstream delta is supplied by
hand, an independently-computed expected gradient contribution.

### 4. `ConvLayer`

Receptive-field index construction (the row-major flat-indexing formula above, with the
dedicated hot-pixel test from "numerical and behavioral risks"), assembling `ConvKernel`s and
`ConvUnit`s, and the five per-layer methods from stage 1's new interface. Testable against a
small synthetic multi-position input (e.g. 4x4 or 5x5) with hand-derived expected outputs for
every position, plus the numerical gradient check from "numerical and behavioral risks" - the
required gate before stage 5.

### 5. `ConvMultiClassBackpropClassifierNetwork`

Constructor (bypassing `MultiClassBackpropClassifierNetwork.__init__`'s uniform
`layer_sizes: list[int]` assumption, building `hidden_layers`/`output_layer`/`trainable_layers`
directly), `randomize()`, and a real `save`/`load` envelope (not `save_model_json`'s existing
one - see "function-level inventory"). End-to-end tested against UCI digits at small scale
first (a handful of epochs, checking training accuracy climbs and doesn't error, before any
real comparison against the 96.9% dense baseline is meaningful).

### 6. the actual validation this workplan exists to produce

Train `ConvMultiClassBackpropClassifierNetwork` on UCI digits (8x8), multiple seeds, compared
directly against `MultiClassBackpropClassifierNetwork`'s existing 96.9% test-accuracy baseline
at the same overall parameter budget as fairly as the two architectures allow - honestly
reporting the result either way, per "numerical and behavioral risks" above. If and only if that
comparison is favorable (or at least not a clear loss) does real MNIST become worth the
wall-clock cost of a full-scale run against the 92.75%/89.12%/96.01% baselines already measured
there.

### 7. what stays explicitly out of scope for this workplan

- **Stacked convolutional layers.** Needs backprop-through-convolution (gradient with respect
  to a conv layer's own input) - real, new math this workplan deliberately avoids needing by
  scoping to exactly one conv layer, directly after the non-trainable input. A follow-on
  workplan of its own, once a single conv layer is proven correct and worth having.
- **Multi-channel input.** Only matters once conv layers stack (this workplan's single conv
  layer sees the raw 1-channel grayscale input directly) - deferred alongside stacking.
- **`'same'` padding.** Needs either synthetic always-zero boundary nodes or special-cased
  edge-position indexing; `'valid'` padding (output shrinks, no padding) is simpler and
  sufficient to prove the core mechanism.
- **Pooling (max or average).** Not needed to prove the core mechanism - a `stride > 1` conv
  achieves comparable downsampling without a new layer type or a max-routing backward pass.
  Worth adding later if a real architecture needs the extra capacity/regularization pooling
  specifically provides, not bundled into "does convolution work at all."
- **Any change to `LinearClassifierNetwork`/`AssociationNode`, or to any existing
  `BackpropLayer` sibling's behavior.** Only the five methods named in stage 1 change, and only
  in ways proven behavior-preserving for every existing layer type first.

## what this document is not

An analysis and a workplan, not an implementation, not a migration plan, and not a decision to
build. Whether to pursue this at all remains the explicitly flagged, undecided question in
[structure](structure.md#possible-next-steps) - the value case here (fewer parameters per
feature, translation invariance) is a real, literature-backed claim, but this codebase's own
practice throughout is to measure it directly against its own real baselines rather than take
it on faith, exactly as stage 6 above is scoped to do.
