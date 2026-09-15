import json


def save_model_json(
    path: str,
    *,
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    class_count: int,
    snapshot: object,
) -> None:
    """
    The shared JSON envelope behind MultiClassBackpropClassifierNetwork.save and
    EnsembleBackpropClassifierNetwork.save - both need exactly enough to reconstruct a
    network's shape (layer_sizes, dimension, input_bounds, class_count) plus its trained
    weights (snapshot); only what goes into snapshot and how many networks get built from this
    envelope differs between the two.
    """

    with open(path, "w") as f:
        json.dump(
            {
                "layer_sizes": layer_sizes,
                "dimension": dimension,
                "input_bounds": input_bounds,
                "class_count": class_count,
                "snapshot": snapshot,
            },
            f,
        )


def load_model_json(path: str) -> dict:
    """
    The load-side counterpart to save_model_json: reads the envelope back, with input_bounds
    already restored to tuples (JSON only has arrays, so a saved (lo, hi) tuple round-trips as a
    2-element list otherwise) - everything else in the envelope is returned as-is for the caller
    to reconstruct its own network shape(s) from.
    """

    with open(path) as f:
        state = json.load(f)

    state["input_bounds"] = [tuple(bound) for bound in state["input_bounds"]]
    return state
