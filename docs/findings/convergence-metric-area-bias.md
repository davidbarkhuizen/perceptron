# The Convergence Disagreement Metric Is Area-Biased Across Cardinality

## Summary

`classification_disagreement_rate()` (`perceptron/train.py`), used throughout the training
loop and both plotting demos to measure how closely a student classifier matches its
reference, samples points **uniformly over the input bounding box** and reports the fraction
where the student disagrees with the reference. This makes it an *area-weighted* metric.
That's a problem, because the reference's positive region (the intersection of `cardinality`
half-planes) shrinks sharply as `cardinality` grows - so the metric increasingly measures "how
well was the large, easy majority region learned" rather than "how well was the classifier
learned," and the two diverge more the higher `cardinality` gets.

The training algorithm itself is unaffected (it learns from individual examples via the
per-example perceptron/minimum-disturbance rule, not from this metric) - this is a
measurement and reporting defect, not a training-correctness defect. It affects the
convergence charts (`demo.py`, `demo_cardinality_sweep.py`) and anything else that reads
`classification_disagreement_rate`'s output as "how converged is the student."

## Background

For a `LinearClassifierNetwork` with `cardinality` hidden nodes, the positive ("class 1")
region is the intersection of `cardinality` half-planes - a convex region whose area shrinks
roughly geometrically as `cardinality` increases, since each additional half-plane can only
shrink (never grow) the intersection. Measured directly (uniformly sampling the bounding box
and checking the reference's own classification):

| cardinality | avg. positive-region share of the bounding box |
|---|---|
| 1 | ~55% |
| 2 | ~28% |
| 3 | ~15% |
| 4 | ~7.5% |

`classification_disagreement_rate(reference, student, sample_count)` draws `sample_count`
points uniformly from `reference.input_bounds` and reports the fraction where
`reference.classify_state(state) != student.classify_state(state)`. Since the samples land in
the positive region only in proportion to its (shrinking) area, disagreement *inside* that
region becomes a smaller and smaller share of the overall reported number as cardinality
grows - it can be almost entirely diluted away by agreement on the majority negative region.

## Evidence

Comparing the existing metric against a class-balanced alternative (equal numbers of
reference-positive and reference-negative samples, drawn by rejection sampling - the same
technique `random_alternating_training_data` already uses to build training sets) on a
**freshly randomized, untrained student** (the "hasn't learned anything yet" baseline):

| cardinality | uniform-sample disagreement | class-balanced disagreement | reality |
|---|---|---|---|
| 1 | 0.371 | 0.360 | roughly matches |
| 2 | 0.231 | 0.595 | student is wrong on 500/500 sampled positive-class points, but the uniform metric reports ~77% agreement |
| 4 | 0.228 | 0.541 | same story - wrong on essentially the whole target region, metric says "looks fine" |

At cardinality 2 and 4, the untrained student has learned **nothing** about the actual target
region, yet the uniform-sampling metric reports it as roughly three-quarters correct. At the
training loop's actual `sample_count=30` per checkpoint, cardinality=4's ~7.5% positive region
gets on average only ~2 samples per checkpoint (often zero) - the metric is largely blind to
that region's error much of the time.

The gap narrows once training has mostly converged - checked with real post-training
students:

| cardinality | uniform (after training) | class-balanced (after training) |
|---|---|---|
| 1 | 0.013 | 0.018 |
| 2 | 0.026 | 0.047 |
| 4 | 0.045 | 0.028 |

- still up to ~1.7x off at cardinality 2 - but the bias is worst exactly when the convergence
chart is most useful: showing what's happening *during* training, not just the two endpoints.

## Impact

- The convergence charts in `demo.py` and `demo_cardinality_sweep.py` understate how much
  the student still has left to learn at higher cardinality, especially early in training,
  and can make training look further along than it is.
- Any future code that gates a decision on `classification_disagreement_rate`'s value (e.g.
  "stop training once disagreement < X") would apply an effectively different, laxer standard
  at higher cardinality than at `cardinality=1`, without that being an intended design choice.

## Suggested fix

Replace (or add alongside) `classification_disagreement_rate` with a class-balanced version -
draw an equal number of reference-positive and reference-negative points via rejection
sampling, and average disagreement equally across both classes:

```python
def class_balanced_disagreement_rate(
    reference: LinearClassifierNetwork,
    student: LinearClassifierNetwork,
    per_class_sample_count: int = 15,
    max_attempts: int = 20_000,
) -> float:

    counts = {0.0: 0, 1.0: 0}
    disagreements = {0.0: 0, 1.0: 0}

    attempts = 0
    while counts[0.0] < per_class_sample_count or counts[1.0] < per_class_sample_count:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"failed to sample {per_class_sample_count} examples of each class within "
                f"{max_attempts} attempts - the reference classifier's decision boundary "
                "likely doesn't cross its input bounds, making one class unreachable"
            )
        attempts += 1

        state = tuple(uniform(*bounds) for bounds in reference.input_bounds)
        reference_category = reference.classify_state(state)
        if counts[reference_category] < per_class_sample_count:
            counts[reference_category] += 1
            if student.classify_state(state) != reference_category:
                disagreements[reference_category] += 1

    return (disagreements[0.0] + disagreements[1.0]) / (counts[0.0] + counts[1.0])
```

This needs the same rejection-sampling infinite-loop guard `random_alternating_training_data`
already has, for the same reason (a reference classifier can make one class unreachable
within its bounds). With this change, "good convergence" means the same thing regardless of
cardinality, instead of an increasingly lenient, area-diluted standard.

## Status

Open - assessed and documented, not yet implemented.
