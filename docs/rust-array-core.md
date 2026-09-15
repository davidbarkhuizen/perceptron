# implementing the required subset in Python-wrapped Rust

[← back to vectorization](vectorization.md)

Part 3 of 3 in the [vectorization](vectorization.md) workplan split: a plan for implementing
[the numpy interface subset](numpy-interface-subset.md)'s exact operation list as a hand-built,
tightly-scoped array core in Rust, wrapped for Python via PyO3/maturin - rather than adopting
real `numpy` as a permanent dependency. Analysis and workplan only - not a decision to build
anything, and nothing in `perceptron/` changes as a result of this document.

## what this replaces, and when

[vectorized array-based model classes](vectorized-array-classes.md) is built and validated
against real `numpy` first, deliberately - see that document's own "why numpy here, now". This
document's array core is the intended eventual replacement for that dependency, not a
parallel or competing effort: once built and proven correct against
[the numpy interface subset](numpy-interface-subset.md)'s exact contract, swapping
`vectorized-array-classes.md`'s classes from `numpy` to this core is the natural follow-on - see
"what stays explicitly out of scope" below for why that swap isn't part of *this* document's own
workplan either.

## why not just keep using real NumPy

Installing `numpy` would obviously work, and would be the pragmatic choice for anyone whose only
goal is faster training. This document exists because this repo's own identity treats even a
"just fast math" dependency as a deliberate choice, not a default - the same posture that's kept
scikit-learn as a one-time, offline, non-runtime extraction tool (`digits_data.py`) rather than a
real dependency, and pyarrow as a one-time conversion step
(`mnist_data.convert_parquet_to_binary`) rather than something training itself ever imports. If a
future maintainer decides plain `numpy` is the right call - including leaving
`vectorized-array-classes.md`'s own classes on it permanently - this entire document is moot.

## decision: a hand-built Rust core (via PyO3/maturin), not C, not real NumPy

Two implementation languages were compared for a from-scratch array core covering exactly
[the numpy interface subset](numpy-interface-subset.md)'s own table (deliberately not
general-purpose NumPy): C wrapped via `ctypes` or the raw CPython C-API, and Rust wrapped via
[PyO3](https://pyo3.rs)/[maturin](https://www.maturin.rs). Rust came out ahead primarily on the
Python-binding layer, not the numerical code itself (the actual math - matmul, elementwise ops,
reductions - is the same algorithmic work in either language): PyO3's
`#[pyclass]`/`#[pyfunction]`/`#[pymodule]` macros generate the reference-counting and
GIL-handling boilerplate automatically, eliminating the single riskiest, hardest-to-debug part of
a raw CPython C-API extension (manual reference-counting bugs - dangling references,
double-frees, refcount leaks), and `maturin` absorbs the packaging/build step a C extension needs
a separate `setup.py`/Makefile for. Real, production-proven precedent exists for exactly this
pattern (`polars`, `ruff`, `orjson`, `cryptography`'s core are all Rust-behind-Python-bindings,
not experiments).

PyO3/Rust is the toolchain this workplan targets; real NumPy remains the pragmatic default this
document is deliberately not recommending over it.

## expected performance

Measured, not assumed - a throwaway benchmark (`numpy` happens to be installed on this machine;
not a repo dependency and not proposed as one here - used purely as a real proxy for "what a
compiled, vectorized implementation could achieve," since it's the closest thing to this
workplan's target already available to measure against) at this codebase's actual architecture
(`dimension=784`, `hidden=16`, `output=10`, matching
`demo_mnist_ensemble_recognition.py`/the softmax investigation). Correctness was checked before
trusting the timing: the pure-Python and vectorized forward passes were run on identical
weights/input and compared (max difference 5.27e-16 - floating-point noise, not a real
discrepancy).

| | time |
|---|---|
| pure Python, per-example forward pass | 1328.1 us |
| vectorized (numpy), per-example forward pass | 8.8 us |
| **speedup** | **150.7x** |
| vectorized, batched (batch_size=8) | 4.74 us/example (amortized) |
| vectorized, batched (batch_size=32) | 3.57 us/example (amortized) |
| vectorized, batched (batch_size=128) | 2.44 us/example (amortized) |

This measures only the forward pass - backward-pass delta computation and gradient application
weren't separately benchmarked here, but are algorithmically the same shape of work
(matrix-vector products, one outer product), so should see a comparable-order speedup; worth its
own direct measurement once/if this work is pursued, not assumed to transfer unchecked.

### relative: numpy is a ceiling, not the actual target

`numpy`'s array operations are backed by OpenBLAS - SIMD instructions and cache-blocked matrix
multiplication, decades of industry tuning. The workplan below deliberately starts with a
**naive** Rust matmul (no SIMD, no blocking) - see "what stays explicitly out of scope." A naive
implementation should be expected to land meaningfully below this 150x ceiling, not match it;
even capturing a fraction of it - 15-45x, say, at 10-30% of the measured numpy speedup - would
still be a large, practically significant win over the current pure-Python baseline. A real PyO3
extension also carries its own Python<->Rust call-marshaling overhead per call, not present in
this pure-Python-vs-numpy proxy, which would eat into the measured ceiling further - another
reason to treat 150x as an upper bound to measure against once a real implementation exists, not
a number to plan around as delivered.

### absolute: what this would mean for this codebase's real, already-measured workloads

Extrapolating the per-example numpy figure above against this session's own real measured
full-scale baseline (the isolated 4.03ms/iteration full `learn()` call - forward + backward +
gradient application - measured for `MultiClassBackpropClassifierNetwork` at this exact
architecture, see [research and analysis](research-and-analysis.md)): a 60000-example epoch
currently takes ~4.0 minutes of pure-Python compute. Even a conservative fraction of the measured
150x forward-pass ceiling (say, 20x, allowing generously for FFI overhead and the naive-vs-BLAS
gap) would put the same epoch under 15 seconds - the kind of change that doesn't just make
existing runs faster, but changes what's practical to run at all: the
[MNIST ensemble](demos.md#demo-mnist-ensemble-recognition)'s ~30-minute real training run,
several-seed statistical sweeps like the ones behind every "measured, not worth adopting" finding
in [research and analysis](research-and-analysis.md), and a proper learning-rate-vs-batch-size
sweep like the one the mini-batch momentum retest's own confound surfaced (see
[structure](structure.md#possible-next-steps)), would each plausibly drop from a real logistical
constraint (worth scheduling, worth running in the background while doing other work - precisely
how every real-MNIST measurement in this session's own history was run) to something fast enough
to iterate on interactively instead.

## workplan

### scope

Fixed `f64` dtype, up to 2D (matrix) + 1D (vector), row-major contiguous storage - exactly
[the numpy interface subset](numpy-interface-subset.md)'s own table, nothing more: array
construction (from data, zero-filled), shape/transpose, single-index write, matmul, elementwise
`+ - * /` with the two scoped broadcasting cases, in-place accumulate, `exp`, `outer`, `argmax`,
a uniform RNG, `.copy()`, raw-byte decode/reshape/slice/cast, and a Python-list round-trip for
serialization.

### 1. crate structure

A new top-level Rust crate (e.g. `rust/perceptron_array/`), kept separate from `perceptron/` and
standalone-buildable until proven correct, not dropped into the existing package tree
prematurely.

- `Cargo.toml`: `pyo3` (with the `extension-module` feature) as the only external crate -
  deliberately no `ndarray`, no `rand` crate, matching this repo's own "hand-build everything"
  posture and answering the same dependency-purity question consistently at this layer too
  (pulling in Rust crates for convenience would just move the same "adopt vs. hand-build" tension
  NumPy itself raises down one level, not resolve it).
- `src/array.rs` - the core type: a plain struct wrapping a flat `Vec<f64>` buffer plus a
  `shape: (usize, usize)` (or an enum distinguishing the 1D/2D cases actually used - no reason to
  build general N-dimensional machinery [the numpy interface subset](numpy-interface-subset.md)
  never needs), with `.copy()`/`.reshape()`/slicing/transpose and single-index read/write.
- `src/ops.rs` - elementwise `+ - * /`, restricted to exactly the two broadcasting cases the
  interface subset names (vector+vector, matrix+row-vector) - not general N-d broadcasting.
- `src/linalg.rs` - matmul (naive triple loop first; a cache-friendlier loop order as a later,
  separately-measured refinement, not bundled into "get it correct" - matches this session's own
  "measure before assuming" posture, applied to performance instead of ML results) and `outer`.
  Not targeting BLAS-competitive performance - that's decades of industry SIMD/cache-tuning work,
  out of scope regardless of language; the realistic target is "meaningfully faster than pure
  Python."
- `src/ufuncs.rs` - `exp`, `argmax` (no axis parameter needed - see
  [the numpy interface subset](numpy-interface-subset.md)'s own "explicitly not required").
- `src/random.rs` - a hand-rolled PRNG (a small, well-known public-domain algorithm - e.g.
  xorshift128+ or PCG32 - implemented directly, not pulled from the `rand` crate) plus a
  uniform-range array-fill function.
- `src/mnist.rs` - raw `u8` buffer -> `f64` array decode (the `/255.0` rescale, reshape, and
  pixel/label slice), a direct port of what `load_mnist_dataset_as_array`
  ([vectorized array-based model classes](vectorized-array-classes.md)) needs.
- `src/lib.rs` - the `#[pymodule]` entry point, exposing an array `#[pyclass]` (a project-specific
  name, avoiding collision with real NumPy's own `PyArray`) with `#[pymethods]` for the operations
  above, plus free `#[pyfunction]`s for the standalone ones (RNG, MNIST decode).

### 2. build integration into this repo's existing tooling

- `cli setup` gains a step installing `rustc`/`cargo` (matching its existing
  `apt install python3-tk` pattern) and `maturin` (via pip, into the same `.venv`).
- `cli setup` (or a new `cli build-rust`) runs `maturin develop` to build the extension into the
  active venv - the Rust-side analogue of `pip install -r requirements.txt`.
- `.gitignore` gains the Rust build output (`rust/*/target/`), matching the existing pattern for
  other regenerable artifacts (`trained_model.json`, `__pycache__`).
- `requirements.txt` stays Python-only; `Cargo.toml` is the Rust-side equivalent, not folded into
  it.

### 3. build order

Each stage independently testable before the next starts - no stage depends on an unvalidated
earlier one:

1. Array construction/shape/indexing only, no math yet - prove the Python<->Rust round-trip
   (construct from a Python list, read values back) works first.
2. Elementwise ops + the two scoped broadcasting cases.
3. `exp` (needed together with the sigmoid-overflow question below).
4. matmul + `outer`.
5. `argmax`, `.copy()`.
6. RNG.
7. MNIST byte decode/reshape/slice/cast.
8. The Python-list round-trip (`.tolist()`-equivalent / construct-from-nested-list), for
   `save()`/`load()`.

### 4. numerical parity validation

For every operation in [the numpy interface subset](numpy-interface-subset.md)'s table: a
randomized-input test comparing the Rust result against **both** real `numpy`'s own result and
the existing pure-Python reference implementation
(`BackpropNode.z()`/`sigmoid()`/[vectorized array-based model classes](vectorized-array-classes.md)'s
own already-numpy-parity-checked methods) across a large random-input sweep - the same discipline
as the 200k-random-z bit-identical sigmoid check already precedented in this codebase, extended to
every operation in the subset rather than assumed to transfer. The two known risk areas
(summation order, sigmoid-overflow behavior) already flagged in
[vectorized array-based model classes](vectorized-array-classes.md)'s own "numerical parity
validation" section need this check explicitly and early, not as an afterthought - checking
against numpy transitively re-validates against the pure-Python reference that document's own
classes were already checked against, so a three-way match (Rust, numpy, pure Python) is strictly
stronger evidence than a two-way one.

### 5. what stays explicitly out of scope for this workplan

- **Swapping [vectorized array-based model classes](vectorized-array-classes.md)'s own classes
  from `numpy` to this core.** That document's classes are built and validated against real numpy
  independently of this one - retargeting their array backend once this core exists and passes
  its own parity checks is a small, mechanical follow-on (an import swap plus a fresh parity
  re-run against the pure-Python reference), but still its own step, not assumed to happen
  automatically or be part of "building the core."
- **BLAS-competitive matmul performance.** Naive-but-correct first; any tuning is a
  separately-measured follow-up, not bundled into "does it work."
- **Any operation outside [the numpy interface subset](numpy-interface-subset.md)'s own table.**
  No axis-parameterized reductions, no `where`/`maximum`/`clip`, no general N-d arrays or
  broadcasting - see that document's own "explicitly not required" for what a future
  ReLU/softmax/momentum/L2 vectorized variant would need to add here first, before this core
  could support it.
- **Any change to `perceptron/model/`.** This document builds and validates a fully standalone
  Rust extension - nothing in the existing pure-Python codebase changes as a result of it.

## what this document is not

An analysis and a workplan, not an implementation, not a migration plan, and not a decision to
build. Whether to pursue this at all - vs. leaving
[vectorized array-based model classes](vectorized-array-classes.md) on real `numpy`
indefinitely, vs. never building those classes at all - remains the explicitly flagged, undecided
question in [structure](structure.md#possible-next-steps): any array-library dependency, hand-built
or adopted, permanent or prototype-only, is a deliberate choice for whoever maintains this repo,
not something to assume is wanted just because it would be faster.
