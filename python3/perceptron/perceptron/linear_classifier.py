from __future__ import annotations

from random import uniform

class SPoint:

    def __init__(self, value: float, min: float = -10.0, max: float = 10.0) -> None:
        self.value = value
        self.min = min
        self.max = max

    def update(self, value: float) -> None:
        self.value = value

    def randomize(self) -> None:
        self.value = uniform(self.min, self.max)

    def output(self) -> float:
        return self.value

class SLayer:

    def __init__(self, values: list[float], bounds: list[tuple[float, float]]) -> None:
        self.s_points = [SPoint(values[i], bounds[i][0], bounds[i][1]) for i in range(len(values))] 

    def size(self):
        return len(self.x)
    
    def update(self, x: list[float]) -> None:
        for i in range(len(self.size())):
            self.x[i].update(x[i])

    def randomize(self) -> None:
        for s_point in self.s_points:
            s_point.randomize()

class Perceptron:
            
    def __init__(self, activation_threshold: float, output_value: float) -> None:

        self.inputs = None        
        self.input_weights = None

        self.activation_threshold = activation_threshold
        self.output_value = output_value

    def connect_inputs(self, inputs: list[SPoint | Perceptron], weights: list[float]):        
        assert(len(inputs) == len(weights))
        
        self.inputs = inputs
        self.input_weights = weights

    def aggregate_input_value(self):
        return sum([self.inputs[x].output() * self.input_weights[x] for x in range(len(self.inputs))])

    def is_active(self) -> bool:
        return self.aggregate_input_value() >= self.activation_threshold

    def output(self) -> float:
        return self.output_value if self.is_active() else 0

def random_input_weights(n: int) -> list[float]:
    return [0.0 for j in range(n)]

def random_activation_threshold() -> float:
    return 1.0

class ALayer:
    
    def __init__(self, units) -> None:
        self.units = units
        
    def size(self):
        return len(self.units)


    def connect_to(self):


    # construct
    # fully connect to
    # - s_layer
    # - parent a_layer


class LinearClassifier:
    pass

    def __init__(self) -> None:

        self.s_layer = SLayer([0.0, 0.0], [(-10, 10), (-10, 10)])
        self.s_layer.randomize()




def entrypoint():

    slayer = SLayer([])    


    n_layer_1 = 4
    layer_1 = [
        Perceptron(i, inputs=s_layer, input_weights=random_input_weights(n_s_layer), activation_threshold=random_activation_threshold(), output_value=1.0) 
        for i in range(n_layer_1)
    ]

    layer_2 = [Perceptron(1, inputs=layer_1, input_weights=[1.0 for i in range(n_layer_1)], activation_threshold=1.0*n_layer_1, output=1.0)]


    # update .value for each SPoint in the s_layer





