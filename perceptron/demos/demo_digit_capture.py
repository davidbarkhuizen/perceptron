import sys
import tkinter as tk

from perceptron.digit_capture import GRID_SIZE, apply_brush_stroke, intensity_to_color, pixel_to_tile, tile_grid_to_state
from perceptron.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork

MODEL_PATH = "data/digits/trained_model.json"
TILE_SIZE = 35
OFF_COLOR = "#000000"


class DigitCaptureApp:
    """
    An 8x8 mouse-painted tile grid, classified live by a MultiClassBackpropClassifierNetwork
    loaded from disk (see demo_digit_recognition.py, which trains and saves it). Tiles hold a
    graded [0.0, 1.0] intensity, not just on/off - each paint stroke sets the touched tile to
    full intensity and softly lights its neighbors too (see digit_capture.apply_brush_stroke),
    approximating the soft, anti-aliased edges the bundled training data's own preprocessing
    produced, rather than the hard single-tile edges a purely binary toggle would give the
    model - a distribution the model was never trained on.
    """

    def __init__(self, root: tk.Tk, classifier: MultiClassBackpropClassifierNetwork) -> None:

        self.classifier = classifier
        self.grid: list[list[float]] = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.tile_ids: list[list[int]] = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

        canvas_size = GRID_SIZE * TILE_SIZE
        self.canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg=OFF_COLOR, highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x0, y0 = col * TILE_SIZE, row * TILE_SIZE
                x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
                self.tile_ids[row][col] = self.canvas.create_rectangle(
                    x0, y0, x1, y1, fill=OFF_COLOR, outline="gray20"
                )

        self.canvas.bind("<Button-1>", self._handle_paint_event)
        self.canvas.bind("<B1-Motion>", self._handle_paint_event)

        self.prediction_label = tk.Label(root, text="draw a digit", font=("TkDefaultFont", 14))
        self.prediction_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        clear_button = tk.Button(root, text="Clear", command=self.clear)
        clear_button.grid(row=2, column=0, columnspan=2, pady=(0, 10))

    def _handle_paint_event(self, event: tk.Event) -> None:
        row, col = pixel_to_tile(event.x, event.y, TILE_SIZE)
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            self._paint_tile(row, col)

    def _paint_tile(self, row: int, col: int) -> None:
        self.grid = apply_brush_stroke(self.grid, row, col)
        self._render_grid()
        self._classify_and_update()

    def _render_grid(self) -> None:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                color = intensity_to_color(self.grid[row][col])
                self.canvas.itemconfig(self.tile_ids[row][col], fill=color)

    def clear(self) -> None:
        self.grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self._render_grid()
        self.prediction_label.config(text="draw a digit")

    def _classify_and_update(self) -> None:
        state = tile_grid_to_state(self.grid)
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
