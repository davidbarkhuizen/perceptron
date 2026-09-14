# research and analysis

[← back to README](../README.md)

Write-ups of investigations that shaped a design decision - the numbers behind a choice, not
just the choice itself. Unlike [structure](structure.md) (what the code does) or
[demos](demos.md) (what each demo shows), this is where the *why*, backed by measurements, lives
once the investigation itself is no longer visible in any single commit.

## parallelizing MNIST training

### context

Training `MultiClassBackpropClassifierNetwork` on the full real MNIST dataset (60000 images,
784 input dimensions, pure Python, no vectorization) was benchmarked directly at ~12.5ms/
iteration with a 30-node hidden layer - about 12.5 minutes/epoch, ~2-2.5 hours for a 10-epoch
run. This machine has 8 CPUs (`nproc` / `os.cpu_count()`, confirmed directly), so the obvious
question was whether `multiprocessing` could cut that by close to a factor of 8.

Two genuinely different parallelization strategies were prototyped and measured against each
other before committing to either.

### approach 1: data-parallel synchronous weight averaging (rejected)

The natural first attempt: shard the training data into 8 pieces, give every worker process a
copy of the *same* network's current weights, have each train independently on its own shard
for one epoch, then average the 8 resulting weight snapshots elementwise and broadcast the
average back out as the next epoch's starting point (a standard "parallel SGD" / federated-
averaging pattern).

Measured on a 2000-image subset, 2 epochs, 8 workers, `[30]`-node hidden layer:

| | sequential (single process) | parallel (8-way, per-epoch averaging) |
|---|---|---|
| wall time | 52.3s | 26.6s |
| train accuracy | 92.7% | 71.3% |
| test accuracy | 85.9% | 63.8% |

Two real problems, not one:

1. **Speedup was only ~2x, not ~8x.** Each epoch re-shipped the *entire* training shard (not
   just the weight snapshot) to every worker over IPC - at MNIST's scale that's roughly 196,000
   floats per shard, every single epoch. This is a fixable inefficiency (give each worker its
   shard once, at pool startup, instead of every round) - but it was never tested further, once
   the second problem made the whole approach unattractive regardless.
2. **Accuracy dropped substantially, not just slightly.** This is a more fundamental issue than
   IPC overhead: averaging the weights of several replicas of a *nonlinear* network is not the
   same as averaging the functions those replicas learned. Each replica in this test only saw
   1/8 of the data for a full epoch before being averaged back together - a much weaker, lossier
   update than true sequential SGD over the same data. The standard mitigation (synchronize
   more often - every few hundred examples instead of once per epoch) trades the accuracy loss
   back for *less* parallel speedup, since more synchronization means more communication
   relative to actual computation. This makes the whole approach a three-way tradeoff between
   speed, accuracy, and sync frequency, with no setting of the dial that's clearly good on all
   three - the fundamental reason it was dropped, not pursued further with a smarter IPC layout.

### approach 2: an ensemble of independently-trained one-vs-rest classifiers (adopted)

A different decomposition sidesteps synchronization entirely, rather than trying to do it more
cheaply. `MultiClassBackpropClassifierNetwork`'s one-vs-rest output layer already means each of
the 10 output nodes is, conceptually, its own binary "is this digit D?" classifier - they just
happen to share one hidden layer today, which is exactly what makes them dependent on each
other during training (a hidden node's gradient depends on *all 10* output deltas).

The proposal: don't share the hidden layer at all. Train 10 completely independent
`BackpropClassifierNetwork`s (already built, unchanged - this is exactly what that class is),
one per digit, each on its own small, class-balanced binary dataset: every real example of
digit D (positive, label 1.0), plus an equal-sized sample drawn evenly across the other 9
digits (negative, label 0.0) - a training set roughly 1/5 the size of the full 60000-image set
per classifier, instead of all 60000. Since nothing is shared between the 10 training runs,
there is no synchronization step of any kind - not "less frequent," genuinely none. This is
about as close to embarrassingly parallel as a learning problem gets.

Measured directly (not estimated) for 3 of the 10 digits (0, 1, 7 - chosen for shape variety:
round, straight, angular), `[16]`-node hidden layer (half the width used above - a binary
sub-problem needs less capacity than full 10-way classification), 5 epochs each:

| digit | balanced training set size | training time | binary training accuracy |
|---|---|---|---|
| 0 | 11846 | 382.2s | 97.79% (still improving) |
| 1 | 13484 | 466.5s | 98.68% (still improving) |
| 7 | 12530 | 435.2s | 96.53% (still improving) |

None of the three had plateaued or converged by epoch 5 - all three were still improving,
meaning even this is a conservative accuracy measurement, not a ceiling.

The one real open question this approach raises - can classifiers trained completely
independently, with no shared representation or joint optimization, actually be compared
fairly against each other via a simple argmax over their probabilities? - was tested directly,
not assumed:

**3-way ensemble argmax accuracy on digits {0, 1, 7}: 98.21%** (614 held-out test samples).

This is comparable to or better than what the shared-hidden-layer approach achieved on the full
10-way problem in earlier testing. Independently-trained probabilities competed fairly.

### honest timing comparison

Per-classifier cost here (6.45ms/iteration, smaller hidden layer, less per-example work) is
roughly half the single shared network's per-iteration cost, but total example-count processed
across all 10 classifiers is *higher* than a single pass over 60000 images (each image is a
positive example for exactly one classifier, and a candidate negative example for several
others) - extrapolating to all 10 digits at 10 epochs each: ~13 minutes/classifier,
~2.15 hours of total sequential-equivalent work. This is not less total computation than
approach 1 or the original single-network plan - it's *more*.

What makes it faster in wall-clock terms is that all 10 jobs are fully independent, so 8
parallel workers clear them in 2 rounds (`ceil(10/8)`) with no communication between rounds:
**~26 minutes wall-clock**, versus ~2-2.5 hours for the original single shared-network run - a
genuine ~5-6x speedup, achieved *alongside* better accuracy rather than trading against it,
which is the opposite tradeoff approach 1 ran into.

### why this worked where approach 1 didn't

Approach 1's bottleneck was never really the IPC inefficiency (that part was fixable) - it was
that the 8 workers' training runs were not actually independent problems, just 8 partial views
of *one* problem, artificially decoupled and then reconciled by averaging. Reconciliation is
where both the communication cost and the accuracy loss came from. Approach 2 removes the need
for reconciliation by making the sub-problems genuinely independent in the first place - 10
different binary questions, not 8 partial answers to the same 10-way question - so there is
nothing to synchronize, communicate, or average, and the parallelism is close to free.

### decision

Proceed with approach 2: a new ensemble class composed of 10 independent
`BackpropClassifierNetwork`s (one per digit), a small balanced-binary-dataset builder, and a
`multiprocessing`-based job dispatcher to train all 10 concurrently. See
[structure](structure.md) for the resulting design once implemented.
