from random import uniform


class StateNode:
    '''
    sense point
    '''

    def __init__(self, bounds: tuple[float, float], value: float = 0.0) -> None:

        self.__value = value
        self.bounds = bounds

    def randomize(self) -> None:
        self.__value = uniform(*self.bounds)

    def value(self) -> float:
        return self.__value

    def update_value(self, value: float) -> None:
        self.__value = value