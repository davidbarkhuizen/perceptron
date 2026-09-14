import pytest

from perceptron.mnist_data import IMAGE_SIZE, load_mnist_dataset


def test_load_mnist_dataset_shape_and_normalization():

    dataset = load_mnist_dataset("data/mnist/mnist-train.parquet", limit=20)

    assert len(dataset) == 20
    assert all(len(state) == IMAGE_SIZE * IMAGE_SIZE for state, _ in dataset)
    assert all(0.0 <= value <= 1.0 for state, _ in dataset for value in state)
    assert all(0 <= label <= 9 for _, label in dataset)


def test_load_mnist_dataset_decodes_a_known_real_sample_correctly():

    # the standard MNIST training set's well-known first-few labels (5, 0, 4, 1, 9, ...) - a
    # sanity check this is genuinely the real dataset, not corrupted or reordered
    dataset = load_mnist_dataset("data/mnist/mnist-train.parquet", limit=5)
    labels = [label for _, label in dataset]
    assert labels == [5, 0, 4, 1, 9]

    # specific pixel values from the first image, read directly off the decoded PNG and
    # independently cross-checked (not re-derived from this loader) before this test was
    # written - chosen to cover every PNG scanline filter type this image's IDAT stream
    # actually uses (confirmed directly: row 0 is None, row 6 is Paeth, row 7 is Sub, row 11
    # is Up), not just coordinates that happen to land on unfiltered rows
    state0, _ = dataset[0]

    def pixel(row: int, col: int) -> float:
        return state0[row * IMAGE_SIZE + col]

    assert pixel(0, 0) == pytest.approx(0 / 255)  # filter type 0 (None)
    assert pixel(6, 10) == pytest.approx(94 / 255)  # filter type 4 (Paeth)
    assert pixel(7, 20) == pytest.approx(82 / 255)  # filter type 1 (Sub)
    assert pixel(11, 11) == pytest.approx(139 / 255)  # filter type 2 (Up)
    assert pixel(11, 13) == pytest.approx(190 / 255)  # filter type 2 (Up)


def test_load_mnist_dataset_test_split_shape():

    dataset = load_mnist_dataset("data/mnist/mnist-test.parquet", limit=10)

    assert len(dataset) == 10
    assert all(len(state) == IMAGE_SIZE * IMAGE_SIZE for state, _ in dataset)


def test_load_mnist_dataset_without_limit_reads_the_full_file():

    # the test split is the smaller of the two bundled files (10000 rows) - still exercises
    # "no limit" without paying the much larger training file's full decode cost
    dataset = load_mnist_dataset("data/mnist/mnist-test.parquet")

    assert len(dataset) == 10000
