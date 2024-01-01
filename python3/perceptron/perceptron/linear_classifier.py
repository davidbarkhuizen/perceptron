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
            
    def __init__(self, inputs, input_weights, activation_threshold, output_value):

        self.inputs = inputs        
        self.input_weights = input_weights
        assert(len(self.input_weights) == len(self.inputs))
        self.input_count = len(self.inputs)

        self.activation_threshold = activation_threshold
        self.output_value = output_value
        self.cached_output = None

    def is_active(self, aggregate_input_value):
        return aggregate_input_value >= self.activation_threshold

    def output(self):

        aggregate_input_value = sum([self.inputs[x].output() * self.input_weights[x] for x in range(self.input_count)])
        
        output = self.output_value if self.is_active(aggregate_input_value) else 0
        self.cached_output = output
        return output

def random_input_weights(n: int) -> list[float]:
    return [0.0 for j in range(n)]

def random_activation_threshold() -> float:
    return 1.0

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





