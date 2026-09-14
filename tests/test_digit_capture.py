import pytest

from perceptron.digit_capture import (
    CENTER_INTENSITY,
    GRID_SIZE,
    MAX_INTENSITY,
    NEIGHBOR_INTENSITY_STEP,
    apply_brush_stroke,
    intensity_to_color,
    pixel_to_tile,
    tile_grid_to_state,
)


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


def test_apply_brush_stroke_sets_center_to_full_intensity_and_softly_lights_neighbors():

    grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    new_grid = apply_brush_stroke(grid, 3, 3)

    assert new_grid[3][3] == CENTER_INTENSITY
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue
            assert new_grid[3 + delta_row][3 + delta_col] == NEIGHBOR_INTENSITY_STEP

    # nothing further away than one tile is touched
    assert new_grid[0][0] == 0.0
    assert new_grid[5][5] == 0.0

    # the original grid is left untouched - apply_brush_stroke returns a new one
    assert grid[3][3] == 0.0
    assert grid[2][2] == 0.0
    assert NEIGHBOR_INTENSITY_STEP < MAX_INTENSITY


def test_apply_brush_stroke_only_lights_in_bounds_neighbors_at_a_corner():

    grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    new_grid = apply_brush_stroke(grid, 0, 0)

    assert new_grid[0][0] == 1.0
    assert new_grid[0][1] == NEIGHBOR_INTENSITY_STEP
    assert new_grid[1][0] == NEIGHBOR_INTENSITY_STEP
    assert new_grid[1][1] == NEIGHBOR_INTENSITY_STEP
    # every other tile (including the three off-grid "neighbors") stays untouched
    assert sum(sum(row) for row in new_grid) == 1.0 + 3 * NEIGHBOR_INTENSITY_STEP


def test_apply_brush_stroke_neighbor_intensity_is_cumulative_and_capped():

    grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    # tile (3, 4) is a neighbor of both (3, 3) and (3, 5) - two separate strokes should add up
    grid = apply_brush_stroke(grid, 3, 3)
    grid = apply_brush_stroke(grid, 3, 5)

    assert grid[3][4] == pytest.approx(2 * NEIGHBOR_INTENSITY_STEP)

    # repeated strokes at the same spot cap at MAX_INTENSITY rather than exceeding it
    for _ in range(10):
        grid = apply_brush_stroke(grid, 3, 3)
    assert grid[3][4] == MAX_INTENSITY


def test_apply_brush_stroke_rejects_out_of_bounds_center():

    grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    with pytest.raises(AssertionError):
        apply_brush_stroke(grid, -1, 0)

    with pytest.raises(AssertionError):
        apply_brush_stroke(grid, 0, GRID_SIZE)


def test_intensity_to_color_maps_to_grayscale_hex():

    assert intensity_to_color(0.0) == "#000000"
    assert intensity_to_color(1.0) == "#ffffff"
    # 0.375 * 255 = 95.625, rounds to 96 = 0x60
    assert intensity_to_color(0.375) == "#606060"


def test_intensity_to_color_rejects_out_of_range_values():

    with pytest.raises(AssertionError):
        intensity_to_color(-0.01)

    with pytest.raises(AssertionError):
        intensity_to_color(1.01)
