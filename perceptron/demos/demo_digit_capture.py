import sys
import tkinter as tk

from perceptron.digit_capture import (
    CAPTURE_GRID_SIZE,
    GRID_SIZE,
    downsample_to_target_grid,
    intensity_to_color,
    pixel_to_tile,
    tile_grid_to_state,
)
from perceptron.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork

MODEL_PATH = "data/digits/trained_model.json"
CAPTURE_TILE_SIZE = 10
PREVIEW_TILE_SIZE = 35
OFF_COLOR = "#000000"
ON_COLOR = "#ffffff"


class DigitCaptureApp:
    """
    Captures a digit the same way the reference work behind the bundled training data did (see
    digit_capture.downsample_to_target_grid): the user paints a binary CAPTURE_GRID_SIZE x
    CAPTURE_GRID_SIZE bitmap by mouse (mirroring NIST's own thresholded pen-on-paper scan),
    which is then genuinely block-counted down to the GRID_SIZE x GRID_SIZE, 0-16-graded shape
    the model was actually trained on - shown live in a second, smaller preview canvas, so
    what the classifier actually sees is visible, not just asserted. Classified live by a
    MultiClassBackpropClassifierNetwork loaded from disk (see demo_digit_recognition.py, which
    trains and saves it) on every stroke.
    """

    def __init__(self, root: tk.Tk, classifier: MultiClassBackpropClassifierNetwork) -> None:

        self.classifier = classifier
        self.capture_grid: list[list[float]] = [[0.0] * CAPTURE_GRID_SIZE for _ in range(CAPTURE_GRID_SIZE)]
        self.capture_tile_ids: list[list[int]] = [[0] * CAPTURE_GRID_SIZE for _ in range(CAPTURE_GRID_SIZE)]
        self.preview_tile_ids: list[list[int]] = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

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

        preview_canvas_size = GRID_SIZE * PREVIEW_TILE_SIZE
        self.preview_canvas = tk.Canvas(
            root, width=preview_canvas_size, height=preview_canvas_size, bg=OFF_COLOR, highlightthickness=0
        )
        self.preview_canvas.grid(row=0, column=1, padx=10, pady=10)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
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
        if self.capture_grid[row][col] == 1.0:
            return
        self.capture_grid[row][col] = 1.0
        self.capture_canvas.itemconfig(self.capture_tile_ids[row][col], fill=ON_COLOR)
        self._update_preview_and_classify()

    def clear(self) -> None:
        for row in range(CAPTURE_GRID_SIZE):
            for col in range(CAPTURE_GRID_SIZE):
                self.capture_grid[row][col] = 0.0
                self.capture_canvas.itemconfig(self.capture_tile_ids[row][col], fill=OFF_COLOR)
        self._update_preview_and_classify()
        self.prediction_label.config(text="draw a digit")

    def _update_preview_and_classify(self) -> None:
        target_grid = downsample_to_target_grid(self.capture_grid)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                color = intensity_to_color(target_grid[row][col])
                self.preview_canvas.itemconfig(self.preview_tile_ids[row][col], fill=color)

        if sum(sum(row) for row in target_grid) == 0.0:
            self.prediction_label.config(text="draw a digit")
            return

        state = tile_grid_to_state(target_grid)
        predicted = self.classifier.classify_state(state)
        probabilities = self.classifier.predict_probabilities(state)
        confidence = probabilities[predicted]
        self.prediction_label.config(text=f"predicted: {predicted}  (confidence {confidence:.2f})")


def main() -> None:

    try:
        classifier = MultiClassBackpropClassifierNetwork.load(MODEL_PATH)
    except FileNotFoundError:
        print(
            f"no trained model found at {MODEL_PATH} - run `./cli demo-digit-recognition` "
            "first to train and save one."
        )
        sys.exit(1)

    root = tk.Tk()
    root.title("digit capture")
    DigitCaptureApp(root, classifier)
    root.mainloop()


if __name__ == "__main__":
    main()
