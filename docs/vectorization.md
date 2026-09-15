# vectorization

[← back to README](../README.md)

Originally analysis and workplan only. Document 1's build has since been greenlit and completed
(numpy-array-backed classes now exist in `perceptron/`, purely additively - see its own "measured
results"); documents 2 and 3 remain analysis and workplan only, not a decision to build. See
"decision: not made here" below for exactly what is, and isn't, decided.
`docs/structure.md`'s "possible next steps" flags NumPy vectorization as a values question
("no ML framework dependency, everything hand-built" is this repo's own stated identity) rather
than a recommendation; this document, and the three it links to below, are the detailed analysis
behind that flag.

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

**Update - real, not extrapolated, measurement now exists.** [Document 1](vectorized-array-classes.md#measured-results)'s
build has been completed and run end to end against real numpy (not yet the Rust core above -
that swap is still undecided): one full real-MNIST epoch (`[30]`-hidden-layer architecture,
batch-size-1 SGD, the practical case, not a forward-pass-only microbenchmark) measured
**33.87x**, landing inside this section's own 15-45x practical-win range. That real run's
pure-Python side took 18.45 minutes (`learn()` plus the same run's own end-of-epoch training-
accuracy pass), not this section's "~4.0 minutes" - flagged here rather than silently left
stale, since re-checking a documented estimate against a real measurement once one exists is
this codebase's own convention (`docs/research-and-analysis.md`'s own ~12.5-minutes/epoch figure
for the identical architecture is closer, though still not identical, likely because it excludes
that same end-of-epoch accuracy pass). The vectorized side measured 32.7s for the whole epoch -
above this section's own "under 15 seconds" extrapolation, but still a 33.87x real speedup, and
still the difference between an epoch fitting inside a coffee break and one taking most of a
half hour.

## decision: not made here

Document 1's own build has since been greenlit and completed (see its "measured results") - that
narrow decision (build and prove out the array-based classes, against real numpy, as a scoped
prototyping vehicle) is no longer open. What's still explicitly undecided: whether `numpy`
becomes more than that prototyping vehicle (a permanent dependency), vs. swapping to
[the Rust core](rust-array-core.md), vs. reverting to pure Python - the question
[structure](structure.md#possible-next-steps) raises for vectorization as a whole. Any of those
is a deliberate choice for whoever maintains this repo, not something to assume is wanted just
because it would be faster.
