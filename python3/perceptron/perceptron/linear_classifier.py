

class SPoint:

    def __init__(self, value):
        self.value = value

    def output(self):
        return self.value


class Perceptron:
            
    def __init__(self, i, inputs, input_weights, activation_threshold, output_value):

        self.i = i

        self.inputs = inputs        
        self.input_weights = input_weights
        assert(len(self.input_weights) == len(self.inputs))
        self.input_count = len(self.inputs)

        self.activation_threshold = activation_threshold
        self.output_value = output_value

    def is_active(self, aggregate_input_value):
        return aggregate_input_value >= self.activation_threshold

    def output(self):

        aggregate_input_value = sum([self.inputs[x].output() * self.input_weights[x] for x in range(self.input_count)])
        
        return self.output_value if self.is_active(aggregate_input_value) else 0

def random_input_weights(n):
    return [0.0 for j in range(n)]

def random_activation_threshold():
    return 1.0

def initial_output_value(layer, n, i):
    return 1.0

def entrypoint():
    
    x = SPoint(0.0)
    y = SPoint(0.0)

    s_layer = list(x, y)
    n_s_layer = len(s_layer)

    n_layer_1 = 4
    layer_1 = [
        Perceptron(i, inputs=s_layer, input_weights=random_input_weights(n_s_layer), activation_threshold=random_activation_threshold(), output_value=1.0) 
        for i in range(n_layer_1)
    ]

    layer_2 = [Perceptron(1, inputs=layer_1, input_weights=[1.0 for i in range(n_layer_1)], activation_threshold=1.0*n_layer_1, output=1.0)