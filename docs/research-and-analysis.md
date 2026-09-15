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

### building it: a real memory-exhaustion failure, and three wrong hypotheses before the right one

Implementing approach 2 for real, at full MNIST scale, hit a genuine failure the small-scale
prototyping above never exercised: the machine (5.7GB RAM, 2GB swap) ran out of memory and
started thrashing hard enough that a training run never completed. Each hypothesis below was
tested directly and ruled out before moving to the next - none were assumed.

**Not a multiprocessing deadlock.** The stuck run's worker processes showed frozen CPU time
(`ps`'s `TIME` column not advancing between checks) and a `futex_wait_queue` wait channel -
consistent with either a genuine deadlock or a process stalled on swapped-out memory.
`free -h` settled it: 1.8GB of the 2GB swap file in use and climbing. This was swap thrashing,
not a hang - confirmed further once killing the stuck processes immediately freed the memory.

**Not `Pool.imap`'s laziness.** A prior fix (see the ensemble-training PRs) built each class's
balanced dataset via a generator passed to `Pool.imap`, intending to keep at most a few datasets
in memory at once rather than all 10 upfront. A minimal repro disproved this: `imap` consumes
its *entire* input generator essentially immediately, regardless of how slowly workers actually
process tasks - confirmed by printing from inside the generator alongside artificially slow
workers and watching all output appear before any worker finished. The "lazy" dataset building
never throttled anything.

**Not `worker_count` being too high.** Capping the worker pool by an estimate of available
memory (not just CPU count) was a real, necessary fix on its own - but reran into the same
thrashing anyway. The estimate itself (based on a small sample's *pickled* size) turned out to
undercount the real cost by roughly an order of magnitude, because it never accounted for what
happens next.

**Not pyarrow's import footprint - genuinely disproven, not just deprioritized.** The next
hypothesis: forking worker processes *after* the main process had already imported `pyarrow`
(needed to load the source parquet files) meant every worker inherited pyarrow's own
multi-hundred-MB footprint, whether or not that worker ever used it. This looked promising and
matched a known Python/multiprocessing pattern - but was tested rather than trusted:

| scenario | worker peak RSS |
|---|---|
| real MNIST data, pyarrow imported in parent, `gc.freeze()` before fork | 2380 MB |
| same, without `gc.freeze()` | 2376 MB (no difference) |
| synthetic data, no pyarrow import anywhere | 803 MB |
| real MNIST data loaded via a new pyarrow-free binary format (`mnist_data.convert_parquet_to_binary`, a one-time offline conversion - kept, since it's useful regardless) | 2254 MB |

The last row is the decisive one: removing pyarrow from the picture entirely barely moved the
number. Whatever was costing ~1.5GB per worker, it wasn't pyarrow.

**The real cause.** The synthetic-data test above used ~11846 freshly-generated examples
directly - a similar count to one class's balanced set, but *not* the product of first loading
all 60000 real examples and then subsampling. The real code path does exactly that: the main
process decodes the *entire* dataset into `(tuple-of-784-floats, label)` pairs before any
subsampling happens, and that fully-decoded structure - 60000 × 784 = 47 million individual
boxed Python float objects - is what gets duplicated into every forked worker. Not a library's
fault: that's just what nearly 47 million Python objects costs, regardless of how they got
there or which library was or wasn't involved in loading them.

**The fix, measured before and after on identical real work:**

| approach | worker peak RSS |
|---|---|
| main process fully decodes all 60000 examples, then ships one class's ~11846-example slice via IPC | 2254.7 MB |
| main process reads only label bytes (cheap); the worker seeks directly to its own ~11846 chosen examples and decodes only those | 374.4 MB |

A ~6x reduction, by ensuring no process - main or worker - ever holds the full dataset decoded
in memory at once. `mnist_data.load_mnist_labels` (label bytes only) and
`load_mnist_records_at_indices` (direct seek, decodes only the requested examples) replace
eager full-dataset decoding; `ensemble_train.select_balanced_indices` decides *which* examples
each class needs from label data alone, and `train_ensemble_parallel_from_indices` has each
worker load only its own selection, itself.

**Verified end to end**, not just at the single-worker/single-class scale above: the real, full
60000-image MNIST training set, all 10 classes, memory-aware worker count (8 workers on this
machine) - **31.2 minutes wall-clock, memory stable throughout (no swap growth), 89.4% held-out
test accuracy**, no digit's confusion-matrix row or column collapsed. The run that previously
couldn't complete at all now finishes reliably, in less time than even the original "ideal"
estimate for this approach.

**The general lesson**, beyond this specific fix: a plausible, mechanism-matching hypothesis
(pyarrow's known import-weight problem, matching a real documented Python gotcha) can still be
wrong. Each hypothesis here was cheap to test in isolation and expensive to have shipped
untested - the pyarrow theory in particular "felt right" and would have been easy to stop
investigating at, had it not been measured directly against a synthetic-data control.

## softmax/cross-entropy re-alignment

### context

An audit of the backprop implementation against canonical literature (Rumelhart/Hinton/
Williams's generalized delta rule; Nielsen's *Neural Networks and Deep Learning*, whose BP1-BP4
equations `backprop_node.py`'s forward/backward math matches essentially exactly - confirmed
both by direct derivation and by this codebase's own independently hand-computed, pinned
regression tests) found the core math correct, but flagged one deliberate deviation:
`MultiClassBackpropClassifierNetwork` treats multi-class classification as one-vs-rest -
`class_count` independent sigmoid output nodes, each trained against a one-hot target with
quadratic (MSE) loss via the same `BackpropNode.compute_output_delta` every binary network uses.
The canonical treatment for a *mutually exclusive* multi-class target (exactly what digit
classification is - one true label per image) is a softmax output layer with cross-entropy loss,
where the delta simplifies to `activation - target` with no extra sigmoid-derivative factor.

Checking every place this one-vs-rest choice is explained (the class's own docstring,
`docs/structure.md`'s existing "multi-class" section) found only incidental reasons -
"`compute_output_delta` needed no changes" - never an argument that one-vs-rest+MSE is the
statistically right model for a mutually-exclusive target. That reads as an unexamined default,
not a considered tradeoff, and was worth fixing.

### why the ensemble is correctly out of scope

`EnsembleBackpropClassifierNetwork`'s own one-vs-rest design (see "parallelizing MNIST
training" above) is a *different*, independently measured tradeoff: removing the shared hidden
layer entirely is what makes its 10 sub-networks genuinely trainable in separate processes with
zero communication. Softmax reintroduces exactly the cross-output coupling that split was built
to eliminate - applying it there would either do nothing (each sub-network has a single output
node; softmax over one logit is a constant 1.0) or require sharing state across independently-
trained processes, undoing the measured design. This re-alignment is scoped to
`MultiClassBackpropClassifierNetwork` (the shared-hidden-layer sibling) only.

### the architectural question, and why it turned out simple

Softmax's cross-node coupling (`a_i = e^z_i / Σ_j e^z_j` needs every sibling's `z_j`) only
affects the **forward** pass. Once every node's activation is computed, softmax+cross-entropy's
delta is `activation - target` - a function of that node's own (already-joint) activation and
its own target alone, exactly as per-node-independent as the one-vs-rest delta it replaces. So
`BackpropNetworkBase`'s backward-pass plumbing (`_backward_hidden_layers`, `_apply_gradients`,
`snapshot`/`restore`) needed zero changes. The only place genuinely needing whole-layer (not
per-node) computation is the output layer's own `forward()`.

That made this purely additive: two small extension points
(`BackpropLayer._node_cls`, `BackpropNetworkBase.output_layer_cls`, both defaulting to the
existing classes - zero behavior change for anything that doesn't override them), a new
`SoftmaxOutputNode`/`SoftmaxOutputLayer` (`perceptron/model/softmax_output_layer.py`), and
`SoftmaxMultiClassBackpropClassifierNetwork` (`perceptron/model/softmax_multiclass_backprop_classifier_network.py`)
- which is a *single* class-attribute override (`output_layer_cls = SoftmaxOutputLayer`) on top
of `MultiClassBackpropClassifierNetwork`, since every other method (`learn`, `_backward`,
`randomize`, `randomized`, `snapshot`, `restore`, `save`, `load`) already worked polymorphically
through `cls(...)`/`self.output_layer_cls` and needed no override at all.

`MultiClassBackpropClassifierNetwork` itself is kept, completely unchanged - existing demos and
already-saved models keep working, and it remains a legitimate, simpler reference point to
contrast the softmax sibling against, the same way this codebase already keeps
`EnsembleBackpropClassifierNetwork`'s differently-motivated design alongside both.

### measured comparison

Same architecture, same seed, same everything except the output layer/loss, trained on the full
bundled UCI digits set (1797 samples, 80/20 split, matching `demo_uci_digit_recognition.py`'s
own 32-node hidden layer / 30 epochs / learning_rate=0.5):

| | training accuracy | best epoch | held-out test accuracy |
|---|---|---|---|
| one-vs-rest (MSE) | 99.51% | 22/30 | 95.54% |
| softmax (cross-entropy) | 100.00% | 11/30 | 96.94% |

Softmax reached full training accuracy in about half the epochs and generalized slightly better
on this run. Not a claim that softmax is *always* better - 1797 samples is small, and one seeded
A/B run isn't a statistically powered comparison - but real, reproducible evidence it isn't worse
here, on top of the semantic correctness win (`predict_probabilities()` now genuinely sums to
1.0, unlike one-vs-rest's independent sigmoids) and avoiding quadratic loss's documented
"learning slowdown" pathology (a confidently-wrong, saturated sigmoid neuron produces a tiny
gradient exactly when the error is largest; cross-entropy's `activation - target` delta doesn't).

### deferred: the binary case

`BackpropClassifierNetwork` (used directly, and inside every `EnsembleBackpropClassifierNetwork`
sub-network) still uses quadratic loss for genuinely binary classification. The canonical fix
there is *binary* cross-entropy, not softmax - it collapses to the same clean `activation -
target` delta for a single sigmoid unit. That's a separate, much larger-blast-radius change
(every binary backprop demo and test in this codebase depends on `BackpropClassifierNetwork`'s
current behavior) and was deliberately left out of this re-alignment.

## binary cross-entropy for BackpropClassifierNetwork

### context

The deferred binary case above was investigated as a follow-up. `BackpropClassifierNetwork` is
directly used by 4 standalone demos (`demo_backprop_circular_boundary.py`,
`demo_backprop_linear_parity_check.py`, `demo_backprop_stripes_architecture_sweep.py`,
`demo_xor_backprop_convergence.py`) and by every one of `EnsembleBackpropClassifierNetwork`'s 10
sub-networks (real MNIST training). Its docstring gives no rationale for quadratic loss beyond
describing the sigmoid/gradient-descent mechanism generically - the same "unexamined default"
pattern the softmax audit found for the multi-class case.

Architecturally, binary cross-entropy is *simpler* to add than softmax was: a single output
node's delta needs nothing from any sibling (unlike softmax's joint normalization), so it
doesn't even need a new `Layer` subclass overriding `forward()` - just a
`CrossEntropyOutputNode(BackpropNode)` overriding `compute_output_delta` to
`self.delta = self.value() - reference_value`, reusing the same `_node_cls`/`output_layer_cls`
hooks the softmax work added.

### measured comparison: cross-entropy is not a free win here

Before assuming the same clean improvement the softmax work found, a cross-entropy variant was
prototyped and measured against `test_backprop_training_pipeline.py`'s exact pinned XOR scenario
(`BackpropClassifierNetwork([8], 2, square_bounds(10.0))`, `learning_rate=1.0`, 100 epochs),
across 10 different (data-generation seed, weight-init seed) pairs:

| seed | quadratic (MSE) | binary cross-entropy |
|---|---|---|
| 0 | 97.00% | 84.67% |
| 1 | 98.33% | 93.67% |
| 2 | 97.33% | 85.67% |
| 3 | 97.33% | 92.33% |
| 4 | 98.33% | 95.00% |
| 5 | 98.33% | 94.00% |
| 6 | 99.33% | 95.67% |
| 7 | 96.67% | 92.00% |
| 8 | 96.67% | 93.67% |
| 9 | 98.67% | 92.00% |
| **mean** | **97.80%** | **91.87%** |

Cross-entropy underperformed quadratic loss on every single seed, not a fluke of one run - the
opposite of what "canonical alignment" would predict as an automatic win, and the opposite of
what the softmax/multi-class comparison above actually found.

### why: cross-entropy needs a smaller learning rate here

Rather than stopping at "cross-entropy is worse" (a plausible-looking but untested conclusion),
the mechanism was tested directly: cross-entropy's delta drops the `a(1-a)` damping term
quadratic loss's delta has, so at a fixed learning rate its effective gradient magnitude is
larger - which can overshoot instead of converging smoothly. Sweeping `learning_rate` down for
the cross-entropy variant, same 10 seeds:

| learning_rate | mean training accuracy |
|---|---|
| 1.0 | 91.87% |
| 0.5 | 95.47% |
| 0.25 | 96.67% |
| 0.1 | 97.60% |

At `learning_rate=0.1`, cross-entropy's mean (97.60%) matches quadratic's mean at its own tuned
`learning_rate=1.0` (97.80%). So cross-entropy isn't worse in principle - the hypothesis holds -
but it is not a drop-in replacement at this codebase's existing, separately-tuned
hyperparameters, unlike softmax's clean win at `demo_uci_digit_recognition.py`'s unmodified
`learning_rate=0.5`.

### why this changes the scope decision from the multi-class case

Every consumer of `BackpropClassifierNetwork` has its own hand-tuned `learning_rate`/`epochs`
(`demo_xor_backprop_convergence.py`'s is pinned into a regression test with exact hand-derived
values), so none of them can simply have the loss function swapped in-place - each would need
its own re-tuning and re-measurement pass. The most consequential case is
`EnsembleBackpropClassifierNetwork`'s real-MNIST training (`ensemble_train.py`,
`learning_rate=0.5`, a real ~31-minute wall-clock run per attempt) - unlike softmax, which was
architecturally excluded there (cross-node coupling would undo the ensemble's parallelization),
binary cross-entropy has no such exclusion, so extending it there is *possible* - but validating
a retuned learning rate on real digit data means multiple expensive real training runs, and this
2D-XOR-toy-problem finding may not even transfer to MNIST's very different input scale/dimension
without its own dedicated measurement.

### decision

Build `BinaryCrossEntropyBackpropClassifierNetwork` as a standalone additive sibling (same
pattern as `SoftmaxMultiClassBackpropClassifierNetwork`: new class, `BackpropClassifierNetwork`
completely untouched, no demo repointed, no existing hyperparameter or pinned test touched) -
it's genuinely useful as a literature-aligned option and cheap/low-risk to add on its own terms.
Do **not** extend it to `EnsembleBackpropClassifierNetwork`/real-MNIST training as part of this
work - that needs its own dedicated learning-rate retuning investigation, given the real cost of
each measurement and the genuine uncertainty (not assumption) about whether the benefit
transfers from this toy problem to that one.

## the ensemble/real-MNIST investigation

The "own dedicated investigation" deferred above was carried out as a follow-up, in three steps,
each cheaper than the last one turned out to be necessary before committing to the next.

### step 1: a cheap prerequisite check found a bigger issue than expected

Before comparing loss functions on the ensemble at all, a first check (no training, seconds to
run): sample real MNIST inputs through a freshly-`randomize()`d `BackpropClassifierNetwork` at
the ensemble's actual architecture (`layer_sizes=[16]`, `dimension=784`) and measure the
resulting pre-activation (`z`) distribution. Result: hidden-layer `z` ranged from **-83 to +87**,
with **83.5% of hidden activations already saturated** (`<0.01` or `>0.99`) *before any training
happens at all*. This is exactly the failure mode `MultiClassBackpropClassifierNetwork.randomize()`'s
own docstring already documented and fixed for that class (fan-in-aware `limit = 1/sqrt(fan_in)`
initialization) - but `BackpropClassifierNetwork`, which every ensemble sub-network uses, still
uses the original per-dimension-bounds-width scaling, tuned for 1-2D geometric problems, never
updated for MNIST's 784-dimension fan-in. This is a confound independent of loss function: any
loss-function comparison run on top of it would be dominated by this pre-existing pathology, not
by cross-entropy vs quadratic loss.

### step 2: a cheap proxy sweep isolating init from loss function

Before spending real training time, a small, fast proxy (320 real MNIST examples, one digit's
class-balanced one-vs-rest binary target, built via the same `select_balanced_indices` machinery
`ensemble_train.py` itself uses, 5 epochs, 5 seeds) tested init scheme and loss function as
separate factors:

| config | mean test accuracy |
|---|---|
| quadratic, current init, learning_rate=0.5 (closest match to current production) | 82.75% |
| quadratic, **fan-in-aware init**, learning_rate=0.5 | **93.75%** |
| cross-entropy, current init, learning_rate=0.5 | 87.50% |
| cross-entropy, fan-in-aware init, learning_rate=0.5 | 92.50% |
| cross-entropy, fan-in-aware init, learning_rate=0.1 | 93.25% |

Fixing initialization alone - holding quadratic loss, the loss function currently in production,
fixed - closed an 11-point gap, consistently across all 5 seeds with no overlap between the two
init schemes' ranges. That is a substantially bigger and more certain effect than the loss
function switch: once init is fixed, quadratic and cross-entropy come out roughly tied.
Cross-entropy's one clear edge in this proxy was *robustness* to the current bad
initialization (87.50% vs 82.75% at current init) - consistent with cross-entropy's gradient not
vanishing when a node is saturated, the same "learning slowdown" mechanism this whole
investigation started from.

### step 3: real, full-scale validation

Two real training runs on the actual 60000-image MNIST training set / 10000-image test set, same
architecture as `demo_mnist_ensemble_recognition.py` (`layer_sizes=[16]`, `learning_rate=0.5`,
`epochs=5`), `seed=0` for reproducibility, monkeypatching `ensemble_train.BackpropClassifierNetwork`
to a fan-in-aware-`randomize()` subclass before training (safe because this pipeline's
multiprocessing is fork-based - forked workers inherit the parent process's already-patched
module state; confirmed directly with a fast, small-slice, `epochs=0` check before committing to
the full run, so the weights in the final ensemble were provably drawn from the patched
`randomize()`, not silently still the default):

| config | wall-clock | test accuracy |
|---|---|---|
| current init, quadratic loss (documented baseline, unseeded) | ~31.2 min | 89.4% |
| **fan-in-aware init, quadratic loss** | 29.6 min | **96.01%** |
| fan-in-aware init, binary cross-entropy loss (learning_rate=0.5, untuned for cross-entropy) | 33.5 min | 92.82% |

The fan-in-aware-init run reached **96.01% test accuracy** - a 6.6-point improvement over the
documented baseline, from an initialization fix alone, at the same wall-clock cost, with every
one of the 10 sub-networks still visibly improving at epoch 5 rather than plateaued (unlike the
proxy's small scale, a real training run can distinguish this properly) - suggesting there may be
more headroom left with more epochs, itself worth a future note.

The cross-entropy run, at the same `learning_rate=0.5` used for quadratic loss (not retuned for
cross-entropy), reached 92.82% - clearly ahead of the *unfixed-init* baseline (89.4%), but 3.19
points behind fan-in-aware-init quadratic loss at the same real scale. Every per-digit binary
training accuracy came in lower than quadratic's own (e.g. digit 9: 94.9% vs 98.3%), the same
pattern the XOR toy problem and the small proxy both already showed: cross-entropy's larger,
undamped gradient needs a smaller learning rate than whatever is tuned for quadratic loss, or it
partially overshoots instead of converging as cleanly. This is now confirmed at all three scales
tested (XOR, small MNIST proxy, full real MNIST) - the same real, replicated effect, not sampling
noise at any one scale.

### interpretation and decision

The investigation's original question - does binary cross-entropy improve the real MNIST
ensemble - gets a clear, three-scale-confirmed answer: **not at the learning rate currently
tuned for quadratic loss**, and retuning it (as the small proxy suggested `learning_rate=0.1`
might, itself unverified at full scale) would cost at least one more ~30+ minute real run to
confirm, for a gain that even the best case seen anywhere in this investigation (the proxy's
93.25%) never exceeded fan-in-aware quadratic loss's real-scale result.

The investigation surfaced a much higher-value, already-confirmed fix instead:
**`BackpropClassifierNetwork`'s initialization scheme**, unrelated to loss function, is the
dominant lever - a 6.6-point real-scale improvement (89.4% -> 96.01%) from applying the exact
same fan-in-aware fix `MultiClassBackpropClassifierNetwork.randomize()` already uses, for the
identical documented reason (fan-in-dependent sigmoid saturation), now confirmed to apply
equally to `BackpropClassifierNetwork` at MNIST's 784-dimension scale. This was not the question
this investigation set out to answer, but the evidence for it is now stronger and cheaper to act
on than the original loss-function question.

**Decision:** this finding - `BackpropClassifierNetwork`'s init scheme needs the same
fan-in-aware fix already applied elsewhere - is significant enough to act on as its own,
separately-scoped piece of work, not folded into this write-up. The binary cross-entropy
question, on the other hand, is considered adequately answered for now: consistently
confirmed not to help at production's current learning rate, across three independent scales,
with no further real-MNIST retuning planned unless a future need specifically calls for it.

## Xavier/Glorot init: measured, not worth adopting

### context

A second backprop-literature audit, after `FanInAwareBackpropClassifierNetwork` was built and
wired into the real MNIST ensemble (see above), asked a follow-up question:
`randomize_fan_in_aware`'s `limit = 1/sqrt(fan_in)` scaling is fan-in-only, and doesn't exactly
match the specific variance target LeCun et al. 1998 derived (`1/fan_in`, which for a uniform
draw needs `limit = sqrt(3)/sqrt(fan_in)`, not `1/sqrt(fan_in)`) - it's closer to a common
practical simplification of that scheme (also, historically, PyTorch's own pre-Kaiming default
`nn.Linear` init). Since every network in this codebase uses sigmoid activation throughout, the
literature's most specifically-tailored scheme for that case is Glorot & Bengio 2010's
("Xavier") initialization, `limit = sqrt(6/(fan_in+fan_out))`, derived to keep both forward
activation variance *and* backward gradient variance stable across layers - not just the forward
term a fan-in-only scheme accounts for. A related, smaller question was whether zero-initializing
biases (the more commonly cited default in the literature, e.g. Goodfellow et al.) would help or
hurt, given the current scheme randomizes biases too.

### measured comparison

Same proxy as the earlier ensemble investigation (320 real MNIST examples, digit 3's
class-balanced one-vs-rest target, 5 epochs, 5 seeds, `learning_rate=0.5` - the demo's own tuned
rate, unchanged, since nothing here is a loss-function change that would need its own retuning).
Three configs: the current production scheme, Xavier/Glorot weights with the same random-scaled
bias, and Xavier/Glorot weights with zero bias:

| config | mean test accuracy |
|---|---|
| (A) current fan-in-only (production) | 93.75% |
| (B) Xavier/Glorot, random bias | 93.75% |
| (C) Xavier/Glorot, zero bias | 93.75% |

A clean null, not a weak signal - (B) and (C) even produced identical per-seed results (93.8%
across all 5 seeds), and (A) landed at the same mean. A cheap prerequisite check (mirroring the
earlier investigation's own step 1) confirmed Xavier/Glorot doesn't reintroduce saturation
either - 0% of hidden activations saturated at MNIST's real 784-dimension scale, matching the
already-fixed fan-in-only scheme exactly.

### interpretation

Unlike the original fan-in-only fix - which showed an unambiguous +11-point effect at this exact
same proxy scale, later confirmed at full real-MNIST scale (+6.6 points, 89.4% -> 96.01%) - this
comparison shows nothing to confirm. The likely reason: once the dominant saturation pathology is
fixed (which both schemes do equally well), the *further* refinement Xavier/Glorot offers over a
simpler fan-in-only scheme - accounting for backward gradient variance via fan_out, not just
forward activation variance via fan_in - mainly matters for deeper networks or unusual
layer-width ratios (the case Glorot & Bengio's own paper studied). This codebase's networks are
shallow (a single hidden layer, in every current use), where that extra term has little room to
matter.

### decision

Not adopted. No real-scale validation run was spent confirming this - the proxy's result is
unambiguous enough (a literal tie across 3 configs x 5 seeds, not a marginal or noisy difference)
that spending ~30 minutes of real training time to re-confirm a null result already this clean
would not be a good use of that time. `randomize_fan_in_aware` and
`FanInAwareBackpropClassifierNetwork` stay as they are; this remains a documented, measured "no"
rather than an untested assumption either way.
