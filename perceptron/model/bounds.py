def validate_input_bounds(dimension: int, input_bounds: list[tuple[float, float]]) -> None:
    """
    Shared precondition behind every model class that takes an explicit input_bounds: one
    (lo, hi) pair per input dimension, each with strictly positive width - both
    LinearClassifierNetwork and BackpropNetworkBase (and so every backprop network class built
    on it) require exactly this.
    """

    assert len(input_bounds) == dimension
    assert all(hi > lo for lo, hi in input_bounds), f"input_bounds must all have positive width; got {input_bounds}"
