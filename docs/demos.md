# demos

[← back to README](../README.md)

## demo

    . cli demo

Runs `perceptron/demos/demo.py`, which replicates `test_training_of_linear_classifier` outside
of pytest: trains a classifier against a random reference classifier and pops up three
windows — the convergence curve on a linear scale, the same curve on a log scale (better for
seeing how fast it converges, since disagreement tends to drop roughly exponentially — see
the cardinality-sweep demo below for more on this), and the decision-boundary chart. Both
convergence charts are smoothed with a trailing moving average. Unlike the test, it isn't
time-boxed — the windows stay open until you close them. It uses `cardinality=4` (four
hyperplanes ANDed together) to also showcase the minimum-disturbance multi-unit learning
rule described in [structure](structure.md); since a higher cardinality shrinks the
reference's positive region, it uses `train.reachable_reference_and_training_data` (also
used by the cardinality-sweep demo) to regenerate the reference rather than risk failing on
one unlucky `randomize()`. A 2D convex region needs at least 3 half-planes to be bounded at
all, so at `cardinality=4` the demo also passes `geometry.is_positive_region_bounded` as that
helper's `is_valid` filter, rejecting (and regenerating) any reference whose positive region
isn't a bounded, closed shape — bounded regions are rare (~10% of random draws at this
cardinality), but rejecting on it is a cheap geometry check with no sampling, so a large
attempt budget is still fast. It also prints the trained student's classification of a fresh
point (never seen during training) alongside the reference's, to show the trained classifier
actually being used to predict, not just compared to the reference by eye on a chart. The
decision-boundary chart's bounds are widened as needed, via `chart.reference_region_bounds`,
so the whole bounded region stays visible instead of being cropped at the training bounds.

## demo: cardinality sweep

    . cli demo-cardinality-sweep

Runs `perceptron/demos/demo_cardinality_sweep.py`, which trains independent reference/student
pairs at `cardinality = 1, 2, 3, 4` and overlays their disagreement-rate convergence curves
on two charts, each built from the series smoothed with a trailing moving average (to see
the trend through the sampling noise from `class_balanced_disagreement_rate`'s small
per-class sample count) — one linear-scale, one log-scale (better for comparing how fast
each cardinality's disagreement rate drops, since it does so roughly exponentially) — and
printing each cardinality's before/after disagreement and a prediction-agreement check. A
disagreement rate of exactly 0.0 has no position on a log axis, so a fully-converged
cardinality's curve on that chart simply stops once it hits zero. Since a higher cardinality
shrinks the reference's positive region (the intersection of more half-planes), some
randomly generated reference
classifiers make one class unreachable within the given bounds — the demo regenerates the
reference (up to 20 times) rather than failing the whole sweep on one unlucky draw.

## demo: unreachable class

    . cli demo-unreachable-class

Runs `perceptron/demos/demo_unreachable_class.py`, a headless, console-only demo of
`random_alternating_training_data`'s safety guard: it first generates training data from a
normal, randomly initialised classifier (retrying if an unlucky `randomize()` happens to make
one class unreachable within the bounds — this alone can occasionally happen), then
deliberately constructs a classifier with tiny weights and a large threshold, whose decision
boundary never crosses its bounds, and shows the resulting `RuntimeError` being raised and
caught instead of hanging forever.

## demo: non-representable target

    . cli demo-nonrepresentable-target

Runs `perceptron/demos/demo_nonrepresentable_target.py`. Every other demo's target is a
`LinearClassifierNetwork` of the same architecture the student trains with, so it's always
representable by construction; this one deliberately isn't. It trains students at
`cardinality = 1..4`, swept across AND, OR, and (where distinct) majority `required_active`
gates (see [structure](structure.md)), against an XOR-style target
(`category = (x > 0) != (y > 0)`, two diagonally opposite quadrants) built as a small
adapter object exposing the same `input_bounds`/`classify_state` interface as
`LinearClassifierNetwork` - which is enough for it to drop straight into
`random_alternating_training_data`/`class_balanced_disagreement_rate` unchanged, no new
training-data or metric code needed. Each configuration's line also reports its
`train_linear_classifier_network` result's `.diagnostic` - whether that run converged,
plateaued, or was still improving (see [structure](structure.md)) - and a summary tally
across all nine closes the demo's own claim that none of them are close to converging,
rather than just asserting it. It prints why: the output layer's weights are always `1.0`
per hidden node, so the output can only be a monotonically non-decreasing function of how
many hidden nodes are active - XOR needs the opposite for some units, which no
`required_active` at any cardinality can express. Finishes by plotting the training data
(colored by true label) against the best-performing student's hyperplanes, to show the
mismatch visually.

## demo: backprop XOR

    . cli demo-backprop-xor

Runs `perceptron/demos/demo_backprop_xor.py` - the direct counterpart to the demo above, using
the exact same `XORTarget` (imported from `demo_nonrepresentable_target.py`, not reimplemented)
and the exact same `random_alternating_training_data`/`train_linear_classifier_network` calls,
but training a `BackpropClassifierNetwork` (see [structure](structure.md#backprop)) instead of a
`LinearClassifierNetwork`. Where every configuration in the previous demo plateaus well short of
convergence, this one reaches a training accuracy well past that ceiling, because its output
layer is trained too - a hidden node here can push the output either way, not just
monotonically increase how likely it is to fire. Prints the same kind of `.diagnostic` summary
as every other training demo, then plots the training data over a probability heatmap
(`chart.plot_classifier_probability_heatmap`) instead of decision lines, since the learned
region isn't a union of half-planes a line can represent.

## demo: backprop architecture sweep

    . cli demo-backprop-architecture-sweep

Runs `perceptron/demos/demo_backprop_architecture_sweep.py`, which mirrors
`demo_cardinality_sweep.py`'s pattern (train several configurations against the same target,
overlay smoothed convergence curves) but compares `BackpropClassifierNetwork` architectures -
`[4]`, `[8]`, `[4, 4]`, `[8, 8]` - instead of `LinearClassifierNetwork` cardinalities. `[8]` and
`[4, 4]` share the same total node count, as do `[4]` and `[8, 8]`'s first layer vs its second,
so the comparison is at a matched node budget rather than just "more capacity wins". The target
is `StripesTarget`, 4 alternating vertical bands - harder than `demo_backprop_xor.py`'s 2-region
XOR, so the architectures actually separate instead of all converging easily. Unseeded, like
every sweep demo in this codebase: which architecture comes out ahead varies noticeably between
runs, since plain fixed-learning-rate gradient descent is sensitive to where random
initialization happens to land - this demo shows that variability rather than asserting depth
or width is definitively better.

## demo: backprop circular target

    . cli demo-backprop-circular-target

Runs `perceptron/demos/demo_backprop_circular_target.py`. Every other demo's target boundary,
representable or not, is built from straight edges - `LinearClassifierNetwork`'s positive
region is always a polygon (an intersection of half-planes, see
`geometry.reference_positive_region_polygon`), so it can only ever facet a curve with more and
shorter edges, never actually curve. This target is a disk (radius 4, centered on the origin) -
a genuinely curved boundary. Trains a single-hidden-layer `BackpropClassifierNetwork` (measured:
reaches ~0.99 training accuracy) and plots the learned probability heatmap against the training
data, visibly showing a smooth, rounded decision boundary rather than a faceted polygon.

## demo: backprop vs. linear

    . cli demo-backprop-vs-linear

Runs `perceptron/demos/demo_backprop_vs_linear.py`. Every other backprop demo picks a target no
`LinearClassifierNetwork` can represent well (XOR, stripes, a circle); this one is the opposite
check - a single half-plane (`cardinality=1`), the easiest possible target and squarely within
`LinearClassifierNetwork`'s own representational sweet spot. Trains a `LinearClassifierNetwork`
and a `BackpropClassifierNetwork` on the exact same reference and training data (measured: both
reach ~0.995-0.999 training accuracy), then plots the reference's boundary (green), the linear
student's boundary (purple), and the backprop student's boundary as a probability heatmap
underneath, all overlaid - a parity check showing backprop learns just as well here, not only on
the harder targets the other backprop demos focus on.

## demo: digit recognition

    . cli demo-digit-recognition

Runs `perceptron/demos/demo_digit_recognition.py` - classic-style handwritten digit recognition,
using `MultiClassBackpropClassifierNetwork` (see [structure](structure.md#multi-class)) instead
of any 2D geometric target. Loads the bundled UCI ML hand-written digits dataset
(`data/digits/digits.csv` via `digits_data.load_digits_dataset`, 8x8 pixel images, 10 classes,
1797 samples), splits 80/20 into train/test, and trains a single-hidden-layer (32-node) network
with `train_linear_classifier_network` - reused completely unmodified, the same as every other
backprop demo. Measured: ~99.5% training accuracy, ~95-97% held-out test accuracy, in under a
minute. Saves the trained model to `data/digits/trained_model.json` (gitignored - a regenerable
build artifact) via the new `save()`/`load()` persistence, printing how to reload it without
retraining. Plots a training-accuracy-by-epoch curve, a confusion matrix
(`chart.plot_confusion_matrix`), and a grid of sample test predictions
(`chart.plot_sample_predictions`) colored to flag any incorrect ones.

## demo: digit capture

    . cli demo-digit-capture

Runs `perceptron/demos/demo_digit_capture.py` - an interactive companion to the digit-recognition
demo above. Loads the model that demo trains and saves (`data/digits/trained_model.json` via
`MultiClassBackpropClassifierNetwork.load`; run `demo-digit-recognition` first if that file
doesn't exist yet), then opens an 8x8 grid of tiles you paint with the mouse (click or
click-and-drag). Every tile you paint reclassifies live, showing the predicted digit and the
model's confidence in it. Unlike every other demo, this one needs a display and mouse input -
it's not part of the automated test suite (`tests/test_digit_capture.py` covers only the pure
grid-flattening/coordinate-mapping/brush logic behind it, not the tkinter UI itself). Painting
is graded, not binary: each stroke sets the touched tile to full intensity and softly lights its
neighbors too (`digit_capture.apply_brush_stroke`), approximating the soft, anti-aliased edges
the bundled training data's own preprocessing produced - measured directly, this closes a real
gap, not just a cosmetic one: the same single-tile-wide vertical stroke that an earlier, purely
binary version of this tool classified as "3" at 0.82 confidence classifies as "1" at 1.00
confidence with graded painting.
