import pytest

from perceptron.digit_capture import GRID_SIZE, pixel_to_tile, tile_grid_to_state


def test_tile_grid_to_state_flattens_row_major():

    grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
    grid[2][3] = 1.0

    state = tile_grid_to_state(grid)

    assert len(state) == 64
    assert state[2 * GRID_SIZE + 3] == 1.0
    assert sum(state) == 1.0


def test_tile_grid_to_state_all_off():

    grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    assert tile_grid_to_state(grid) == tuple(0.0 for _ in range(64))


def test_tile_grid_to_state_all_on():

    grid = [[1.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    assert tile_grid_to_state(grid) == tuple(1.0 for _ in range(64))


def test_tile_grid_to_state_rejects_wrong_row_count():

    grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE - 1)]

    with pytest.raises(AssertionError):
        tile_grid_to_state(grid)


def test_tile_grid_to_state_rejects_wrong_column_count():

    grid = [[0.0] * (GRID_SIZE - 1) for _ in range(GRID_SIZE)]

    with pytest.raises(AssertionError):
        tile_grid_to_state(grid)


def test_pixel_to_tile_maps_coordinates_to_the_containing_tile():

    tile_size = 35

    assert pixel_to_tile(0, 0, tile_size) == (0, 0)
    assert pixel_to_tile(34, 34, tile_size) == (0, 0)
    assert pixel_to_tile(35, 0, tile_size) == (0, 1)
    assert pixel_to_tile(0, 35, tile_size) == (1, 0)
    assert pixel_to_tile(35 * 7 + 5, 35 * 7 + 5, tile_size) == (7, 7)


def test_pixel_to_tile_does_not_clamp_out_of_range_coordinates():

    # the caller's job to reject these, not pixel_to_tile's - see demo_digit_capture.py
    assert pixel_to_tile(-1, -1, 35) == (-1, -1)
    assert pixel_to_tile(35 * 8, 0, 35) == (0, 8)
