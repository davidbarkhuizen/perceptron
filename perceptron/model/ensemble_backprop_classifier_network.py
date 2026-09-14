from __future__ import annotations

import json

from perceptron.model.backprop_classifier_network import BackpropClassifierNetwork


class EnsembleBackpropClassifierNetwork:
    """
    A multi-class classifier composed of class_count completely independent
    BackpropClassifierNetworks (already built, unchanged - each one already is a single binary
    classifier with its own hidden layer), one per class, each trained on its own "is this class
    C?" binary problem with no shared state between them at all.

    Unlike MultiClassBackpropClassifierNetwork (one shared hidden layer, class_count output
    nodes, jointly trained - kept as-is, unaffected by this class), there is nothing to
    synchronize during training here: every sub-network is trained completely independently
    (see perceptron/ensemble_train.py), which is what makes training genuinely, not just
    approximately, parallelizable across processes - see docs/research-and-analysis.md's
    "parallelizing MNIST training" entry for the measurements behind this design.

    Unlike the other model classes in this codebase, __init__ doesn't build its own sub-networks
    from layer_sizes/dimension/input_bounds - it just assembles already-constructed
    BackpropClassifierNetworks, since the real use cases are "assemble from independently,
    (often already parallel-)trained classifiers" or "reconstruct from a loaded snapshot", not
    "build one fresh and train it as a whole" (there is no "as a whole" training step for this
    class - see ensemble_train.train_ensemble_parallel).
    """

    def __init__(self, classifiers: list[BackpropClassifierNetwork]) -> None:
        assert len(classifiers) >= 2, f"an ensemble needs at least 2 classifiers; got {len(classifiers)}"
        self.classifiers = classifiers
        self.class_count = len(classifiers)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return [classifier.predict_probability(state) for classifier in self.classifiers]

    def classify_state(self, state: tuple[float, ...]) -> int:
        probabilities = self.predict_probabilities(state)
        return max(range(self.class_count), key=lambda i: probabilities[i])

    def snapshot(self) -> list[list[list[tuple[list[float], float]]]]:
        return [classifier.snapshot() for classifier in self.classifiers]

    def restore(self, snapshot: list[list[list[tuple[list[float], float]]]]) -> None:
        for classifier, classifier_snapshot in zip(self.classifiers, snapshot):
            classifier.restore(classifier_snapshot)

    def save(self, path: str) -> None:
        assert len({classifier.dimension for classifier in self.classifiers}) == 1, "every classifier must share a dimension"
        first = self.classifiers[0]
        with open(path, "w") as f:
            json.dump(
                {
                    "class_count": self.class_count,
                    "layer_sizes": [layer.size for layer in first.hidden_layers],
                    "dimension": first.dimension,
                    "input_bounds": first.input_bounds,
                    "snapshot": self.snapshot(),
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "EnsembleBackpropClassifierNetwork":
        with open(path) as f:
            state = json.load(f)

        dimension = state["dimension"]
        input_bounds = [tuple(bound) for bound in state["input_bounds"]]
        layer_sizes = state["layer_sizes"]

        classifiers = [BackpropClassifierNetwork(layer_sizes, dimension, input_bounds) for _ in range(state["class_count"])]
        ensemble = cls(classifiers)
        ensemble.restore(state["snapshot"])
        return ensemble
