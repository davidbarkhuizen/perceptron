from typing import Any

class SPoint:
    def __init__(self, x: int, y: int, state: bool = False) -> None:
        self.x = x
        self.y = y
        self.state = state

class SSet:
    def __init__(self, points: list[SPoint]) -> None:
        self.points = points
            
class SSystem:
    pass

class AUnit:
    def __init__(self, Q, positive_sense_points, negative_sense_points, v):
        self.Q = Q
        self.positive_sense_points = positive_sense_points
        self.negative_sense_points = negative_sense_points
        self.v = v

class ASet:
    def __init__(self, units, r_set):
        self.unit = units
        self.r_set = r_set

class ASystem:
    pass

class RUnit:
    pass

class RSet:
    def __init__(self, units):
        self.unit = units

class RSystem:
    pass

description = 'perceptron'
print(description)