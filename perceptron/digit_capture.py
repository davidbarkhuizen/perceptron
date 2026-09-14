GRID_SIZE = 8

# a full-strength paint on one tile, in the same [0.0, 1.0]-normalized scale
# digits_data.load_digits_dataset uses (matching the training data's 0-16 grading, so 1.0
# here corresponds to the maximum pixel value the model was actually trained on)
CENTER_INTENSITY = 1.0

# how much a painted tile's orthogonal/diagonal neighbors are nudged up per stroke - 6/16 of
# full intensity, cumulative across repeated or nearby strokes (capped at MAX_INTENSITY) -
# approximates a real pen stroke's soft edges spilling partially into adjacent training-data
# cells, rather than the hard single-tile edges a purely binary toggle produces
NEIGHBOR_INTENSITY_STEP = 0.375

MAX_INTENSITY = 1.0


def tile_grid_to_state(grid: list[list[float]]) -> tuple[float, ...]:
    """
    Flattens an 8x8 tile grid into the same 64-value, row-major, [0.0, 1.0]-normalized state
    shape digits_data.load_digits_dataset produces (row 0 -> indices 0-7, row 1 -> 8-15, ...) -
    matching exactly how sklearn.datasets.load_digits().data flattens its own 8x8 images, since
    that's what data/digits/digits.csv was extracted from. Kept separate from the interactive
    capture UI (perceptron/demos/demo_digit_capture.py) because it's the one piece of that
    tool's logic that's actually a pure function, and so the one piece worth unit-testing the
    same way as everything else in this codebase - the tkinter mouse/canvas code itself isn't
    meaningfully testable without a real or virtual display.
    """

    assert len(grid) == GRID_SIZE, f"grid must have {GRID_SIZE} rows; got {len(grid)}"
    assert all(len(row) == GRID_SIZE for row in grid), f"every row must have {GRID_SIZE} columns"

    return tuple(value for row in grid for value in row)


def pixel_to_tile(x: int, y: int, tile_size: int) -> tuple[int, int]:
    """
    Maps a canvas-relative pixel coordinate to the (row, col) tile it falls in - simple integer
    division, since every tile is the same fixed size. Doesn't clamp/validate against
    GRID_SIZE - a mouse-drag event firing just outside the canvas can produce a negative or
    too-large row/col, and it's the caller's job to decide whether to ignore that (see
    demo_digit_capture.py's _paint_at), not this function's.
    """

    return y // tile_size, x // tile_size


def apply_brush_stroke(grid: list[list[float]], row: int, col: int) -> list[list[float]]:
    """
    Returns a new grid (grid itself is left untouched) with a soft-edged brush stroke applied
    at (row, col): the touched tile jumps straight to full intensity (one click reliably makes
    a visible mark - the tool stays usable rather than needing many passes just to see
    anything), while its orthogonal/diagonal neighbors are nudged up by NEIGHBOR_INTENSITY_STEP
    (capped at MAX_INTENSITY, and cumulative - repeated or nearby strokes keep adding to a
    neighbor's intensity rather than overwriting it).

    This approximates how a real pen stroke actually produced the bundled training data's 0-16
    grading in the first place: NIST's own preprocessing averaged a higher-resolution scan down
    into 8x8 cells, so a stroke passing near a cell's edge left it partially, not fully, lit.
    A purely binary toggle (this tool's first cut) can't produce that - every tile was either
    fully on or fully off, with hard edges no real training example has.
    """

    assert len(grid) == GRID_SIZE and all(len(r) == GRID_SIZE for r in grid), "grid must be GRID_SIZE x GRID_SIZE"
    assert 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE, f"(row, col) must be within the grid; got ({row}, {col})"

    new_grid = [list(r) for r in grid]
    new_grid[row][col] = CENTER_INTENSITY

    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue
            neighbor_row, neighbor_col = row + delta_row, col + delta_col
            if 0 <= neighbor_row < GRID_SIZE and 0 <= neighbor_col < GRID_SIZE:
                new_grid[neighbor_row][neighbor_col] = min(
                    MAX_INTENSITY, new_grid[neighbor_row][neighbor_col] + NEIGHBOR_INTENSITY_STEP
                )

    return new_grid


def intensity_to_color(intensity: float) -> str:
    """
    Maps a [0.0, 1.0]-normalized tile intensity to a grayscale hex color for rendering
    (0.0 -> "#000000", 1.0 -> "#ffffff") - the same linear scale digits_data.py's
    normalization uses, just inverted back to a displayable 0-255 color channel.
    """

    assert 0.0 <= intensity <= 1.0, f"intensity must be in [0.0, 1.0]; got {intensity}"

    level = round(intensity * 255)
    return f"#{level:02x}{level:02x}{level:02x}"
