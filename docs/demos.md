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
training-data or metric code needed. None of the nine configurations get close to
converging, and prints why: the output layer's weights are always `1.0` per hidden node, so
the output can only be a monotonically non-decreasing function of how many hidden nodes are
active - XOR needs the opposite for some units, which no `required_active` at any cardinality
can express. Finishes by plotting the training data (colored by true label) against the
best-performing student's hyperplanes, to show the mismatch visually.
