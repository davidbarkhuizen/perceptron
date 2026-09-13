GRID_SIZE = 8


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
