import random

import pytest

from perceptron.ensemble_train import build_balanced_binary_dataset


def _synthetic_dataset(counts: dict[int, int]) -> list[tuple[tuple[float, ...], int]]:
    # each example's state is a distinct singleton tuple, so identity is easy to trace
    dataset = []
    for label, count in counts.items():
        for i in range(count):
            dataset.append(((float(label), float(i)), label))
    return dataset


def test_positives_are_exactly_every_example_of_the_target_label():

    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})

    result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    positives = [state for state, category in result if category == 1.0]
    assert len(positives) == 6
    assert set(positives) == {(0.0, float(i)) for i in range(6)}


def test_negatives_are_evenly_stratified_across_other_classes():

    # 6 positives, 3 other classes, each with plenty available - divides evenly, so this
    # should be exactly 2 from each other class, not just 6 total drawn from a pooled sample
    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})

    result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    negatives = [state for state, category in result if category == 0.0]
    assert len(negatives) == 6
    for other_label in (1, 2, 3):
        count_from_label = sum(1 for state in negatives if state[0] == float(other_label))
        assert count_from_label == 2


def test_negative_remainder_is_spread_across_the_first_few_classes():

    # 7 positives, 3 other classes: base_count=2, remainder=1 - the first class (sorted order)
    # gets 3, the other two get 2 each, totaling 7
    dataset = _synthetic_dataset({0: 7, 1: 10, 2: 10, 3: 10})

    result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    negatives = [state for state, category in result if category == 0.0]
    counts_by_label = {
        other_label: sum(1 for state in negatives if state[0] == float(other_label)) for other_label in (1, 2, 3)
    }
    assert sorted(counts_by_label.values()) == [2, 2, 3]
    assert sum(counts_by_label.values()) == 7


def test_negative_sampling_is_capped_by_availability():

    # class 1 only has 1 example available, far fewer than the 3 that would otherwise be drawn
    # from it (9 positives / 3 other classes = 3 each) - it should contribute just that 1, not
    # raise or pad, and the total negative count falls correspondingly short of 9
    dataset = _synthetic_dataset({0: 9, 1: 1, 2: 10, 3: 10})

    result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    negatives = [state for state, category in result if category == 0.0]
    count_from_label_1 = sum(1 for state in negatives if state[0] == 1.0)
    assert count_from_label_1 == 1
    assert len(negatives) == 1 + 3 + 3  # short of the full 9 by class 1's 2-example shortfall


def test_result_is_shuffled_and_labels_are_recoded_to_floats():

    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})

    result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    assert len(result) == 12
    assert all(category in (0.0, 1.0) for _, category in result)
    # not simply "all positives first, then all negatives" - a real (if probabilistic) shuffle
    # check: the first half isn't all one category
    categories_in_order = [category for _, category in result]
    assert len(set(categories_in_order[:6])) == 2


def test_reproducible_under_a_fixed_seed():

    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})

    result_a = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(42))
    result_b = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(42))

    assert result_a == result_b


def test_rejects_an_out_of_range_target_label():

    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})

    with pytest.raises(AssertionError):
        build_balanced_binary_dataset(dataset, target_label=4, class_count=4, rng=random.Random(0))
