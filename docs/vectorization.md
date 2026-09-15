# vectorization

[← back to README](../README.md)

Analysis and workplan only - not a decision to build anything here, and nothing in
`perceptron/` changes as a result of this document. `docs/structure.md`'s "possible next
steps" flags NumPy vectorization as a values question ("no ML framework dependency,
everything hand-built" is this repo's own stated identity) rather than a recommendation; this
document, and the three it links to below, are the detailed analysis behind that flag.

## why not just adopt real NumPy

Installing `numpy` would obviously work, and would be the pragmatic choice for anyone whose
only goal is faster training. This document exists because this repo's own identity treats
even a "just fast math" dependency as a deliberate choice, not a default - the same posture
that's kept scikit-learn as a one-time, offline, non-runtime extraction tool
(`digits_data.py`) rather than a real dependency, and pyarrow as a one-time conversion step
(`mnist_data.convert_parquet_to_binary`) rather than something training itself ever imports.
If a future maintainer decides plain `numpy` is the right call, everything below - and the
Rust alternative it plans - is moot. It's written up because the alternative was asked to be
scoped in real detail, not because hand-rolling it is being recommended over the obvious
off-the-shelf option.

## how this is organized

Originally one document; split into three, each answering a different question, in the order
they'd actually need building:

1. **[vectorized array-based model classes](vectorized-array-classes.md)** - what new model
   classes would look like if this codebase's forward/backward/gradient math were rewritten
   around whole-layer arrays instead of individual node objects, built and validated against
   real `numpy` first (a deliberate, scoped prototyping choice - not a decision to adopt it
   permanently, see that document's own "why numpy here, now"). Covers the central
   architectural finding this whole effort turns on: an array-based network can't reuse
   `BackpropNetworkBase`'s existing per-node orchestration at all, so it has to be a genuinely
   standalone class, sharing only the external contract (`learn`, `classify_state`, `snapshot`,
   ...) every other sibling network in this codebase already shares.
2. **[the numpy interface subset](numpy-interface-subset.md)** - the precise, minimal set of
   numpy operations that document's classes actually call, derived directly from their design -
   not a speculative general checklist. This is the formal contract the third document needs to
   satisfy.
3. **[implementing the required subset in Python-wrapped Rust](rust-array-core.md)** - a
   hand-built, tightly-scoped array core in Rust, wrapped for Python via PyO3/maturin,
   implementing exactly document 2's contract - the eventual replacement for document 1's numpy
   dependency, once built and proven correct against it.

Each is independently useful groundwork and independently gated on the one before it: document
1 can be built and validated without document 3 existing yet (that's the whole point of
prototyping against real numpy first); document 3 can't be scoped precisely without document
1's classes existing to derive document 2's contract from. None of the three implies the others
must be built - see each document's own "what stays explicitly out of scope" and "what this
document is not" for exactly where each one's boundary sits.

## expected effect, in one place

Measured, not assumed (see [the Rust implementation plan](rust-array-core.md#expected-performance)
for the full benchmark and its methodology): a throwaway numpy benchmark at this codebase's real
architecture measured **150.7x** forward-pass speedup over the current pure-Python
implementation, correctness-checked first (max difference 5.27e-16 against the pure-Python
reference). Treated as a ceiling, not a target - a naive, unoptimized Rust core should land
well below it, with 15-45x (10-30% of the measured ceiling) still a large, practically
significant win: a 60000-example MNIST epoch's current ~4.0 minutes of pure-Python compute
would plausibly drop under 15 seconds even at the conservative end of that range, changing what's
practical to run at all - real-scale sweeps like the ones behind every "measured, not worth
adopting" finding in [research and analysis](research-and-analysis.md), or the
learning-rate-vs-batch-size follow-up the mini-batch momentum retest's own confound surfaced
(see [structure](structure.md#possible-next-steps)), from a logistical constraint worth
scheduling to something fast enough to iterate on interactively.

## decision: not made here

Whether to pursue any of this - vs. plain `numpy` (permanently, not just as document 1's
prototyping vehicle), vs. leaving the codebase pure Python - remains the explicitly flagged,
undecided question in [structure](structure.md#possible-next-steps): any array-library
dependency, hand-built or adopted, temporary or permanent, is a deliberate choice for whoever
maintains this repo, not something to assume is wanted just because it would be faster.
