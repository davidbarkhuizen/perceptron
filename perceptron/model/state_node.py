from random import uniform

from perceptron.model.base_node import AbstractNode


class StateNode(AbstractNode):
    """
    sense point
    """

    def __init__(self, bounds: tuple[float, float], value: float = 0.0) -> None:

        self.__value: float = value
        self.bounds: tuple[float, float] = bounds

    def randomize(self) -> None:
        self.__value = uniform(*self.bounds)

    def value(self) -> float:
        return self.__value

    def update_value(self, value: float) -> None:
        self.__value = value
