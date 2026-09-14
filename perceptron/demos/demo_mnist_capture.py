import sys
import tkinter as tk

from perceptron.digit_capture import intensity_to_color, pixel_to_tile
from perceptron.mnist_capture import (
    CANVAS_SIZE,
    CAPTURE_BRUSH_RADIUS,
    CAPTURE_GRID_SIZE,
    paint_brush_stroke,
    preprocess_capture,
)
from perceptron.model.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork

MODEL_PATH = "data/mnist/trained_model.json"
CAPTURE_TILE_SIZE = 8
PREVIEW_TILE_SIZE = 10
OFF_COLOR = "#000000"
ON_COLOR = "#ffffff"


class MnistCaptureApp:
    """
    Captures a digit the same way real MNIST source data was itself produced (see
    mnist_capture.preprocess_capture): the user paints a binary CAPTURE_GRID_SIZE x
    CAPTURE_GRID_SIZE bitmap by mouse (each stroke stamped CAPTURE_BRUSH_RADIUS cells wide, via
    mnist_capture.paint_brush_stroke - a single-cell-wide mouse line is far thinner than any real
    digit stroke once cropped and scaled down), which is then genuinely cropped to its own
    bounding box, aspect-preserving-anti-aliased-scaled to fit a 20px box, and center-of-mass-
    placed into a 28x28 field - shown live in a second, smaller preview canvas, so what the
    classifier actually sees is visible, not just asserted. Classified live by an
    EnsembleBackpropClassifierNetwork loaded from disk (see demo_mnist_recognition.py, which
    trains and saves it) on every stroke.
    """

    def __init__(self, root: tk.Tk, classifier: EnsembleBackpropClassifierNetwork) -> None:

        self.classifier = classifier
        self.capture_grid: list[list[float]] = [[0.0] * CAPTURE_GRID_SIZE for _ in range(CAPTURE_GRID_SIZE)]
        self.capture_tile_ids: list[list[int]] = [[0] * CAPTURE_GRID_SIZE for _ in range(CAPTURE_GRID_SIZE)]
        self.preview_tile_ids: list[list[int]] = [[0] * CANVAS_SIZE for _ in range(CANVAS_SIZE)]

        capture_canvas_size = CAPTURE_GRID_SIZE * CAPTURE_TILE_SIZE
        self.capture_canvas = tk.Canvas(
            root, width=capture_canvas_size, height=capture_canvas_size, bg=OFF_COLOR, highlightthickness=0
        )
        self.capture_canvas.grid(row=0, column=0, padx=10, pady=10)

        for row in range(CAPTURE_GRID_SIZE):
            for col in range(CAPTURE_GRID_SIZE):
                x0, y0 = col * CAPTURE_TILE_SIZE, row * CAPTURE_TILE_SIZE
                x1, y1 = x0 + CAPTURE_TILE_SIZE, y0 + CAPTURE_TILE_SIZE
                self.capture_tile_ids[row][col] = self.capture_canvas.create_rectangle(
                    x0, y0, x1, y1, fill=OFF_COLOR, outline=""
                )

        self.capture_canvas.bind("<Button-1>", self._handle_paint_event)
        self.capture_canvas.bind("<B1-Motion>", self._handle_paint_event)

        preview_canvas_size = CANVAS_SIZE * PREVIEW_TILE_SIZE
        self.preview_canvas = tk.Canvas(
            root, width=preview_canvas_size, height=preview_canvas_size, bg=OFF_COLOR, highlightthickness=0
        )
        self.preview_canvas.grid(row=0, column=1, padx=10, pady=10)

        for row in range(CANVAS_SIZE):
            for col in range(CANVAS_SIZE):
                x0, y0 = col * PREVIEW_TILE_SIZE, row * PREVIEW_TILE_SIZE
                x1, y1 = x0 + PREVIEW_TILE_SIZE, y0 + PREVIEW_TILE_SIZE
                self.preview_tile_ids[row][col] = self.preview_canvas.create_rectangle(
                    x0, y0, x1, y1, fill=OFF_COLOR, outline="gray20"
                )

        self.prediction_label = tk.Label(root, text="draw a digit", font=("TkDefaultFont", 14))
        self.prediction_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        clear_button = tk.Button(root, text="Clear", command=self.clear)
        clear_button.grid(row=2, column=0, columnspan=2, pady=(0, 10))

    def _handle_paint_event(self, event: tk.Event) -> None:
        row, col = pixel_to_tile(event.x, event.y, CAPTURE_TILE_SIZE)
        if 0 <= row < CAPTURE_GRID_SIZE and 0 <= col < CAPTURE_GRID_SIZE:
            self._paint_capture_tile(row, col)

    def _paint_capture_tile(self, row: int, col: int) -> None:
        self.capture_grid = paint_brush_stroke(self.capture_grid, row, col)

        # only the brush's own bounding box could have changed - cheap enough to just
        # re-render every tile in it, clipped to the grid, rather than diffing
        for brush_row in range(max(0, row - CAPTURE_BRUSH_RADIUS), min(CAPTURE_GRID_SIZE, row + CAPTURE_BRUSH_RADIUS + 1)):
            for brush_col in range(
                max(0, col - CAPTURE_BRUSH_RADIUS), min(CAPTURE_GRID_SIZE, col + CAPTURE_BRUSH_RADIUS + 1)
            ):
                self.capture_canvas.itemconfig(self.capture_tile_ids[brush_row][brush_col], fill=ON_COLOR)

        self._update_preview_and_classify()

    def clear(self) -> None:
        for row in range(CAPTURE_GRID_SIZE):
            for col in range(CAPTURE_GRID_SIZE):
                self.capture_grid[row][col] = 0.0
                self.capture_canvas.itemconfig(self.capture_tile_ids[row][col], fill=OFF_COLOR)
        self._update_preview_and_classify()
        self.prediction_label.config(text="draw a digit")

    def _update_preview_and_classify(self) -> None:
        target_grid = preprocess_capture(self.capture_grid)

        for row in range(CANVAS_SIZE):
            for col in range(CANVAS_SIZE):
                color = intensity_to_color(target_grid[row][col])
                self.preview_canvas.itemconfig(self.preview_tile_ids[row][col], fill=color)

        if sum(sum(row) for row in target_grid) == 0.0:
            self.prediction_label.config(text="draw a digit")
            return

        state = tuple(value for row in target_grid for value in row)
        predicted = self.classifier.classify_state(state)
        probabilities = self.classifier.predict_probabilities(state)
        confidence = probabilities[predicted]
        self.prediction_label.config(text=f"predicted: {predicted}  (confidence {confidence:.2f})")


def main() -> None:

    try:
        classifier = EnsembleBackpropClassifierNetwork.load(MODEL_PATH)
    except FileNotFoundError:
        print(
            f"no trained model found at {MODEL_PATH} - run `./cli demo-mnist-recognition` "
            "first to train and save one."
        )
        sys.exit(1)

    root = tk.Tk()
    root.title("MNIST capture")
    MnistCaptureApp(root, classifier)
    root.mainloop()


if __name__ == "__main__":
    main()
