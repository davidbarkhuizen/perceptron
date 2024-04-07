from __future__ import annotations

class SPoint:

    def __init__(self, value: float, bounds: tuple[float, float]) -> None:

        self.value = value
        self.bounds = bounds
        
    def update(self, value: float) -> None:
        
        self.value = value

    def randomize(self) -> None:
        
        self.value = uniform(*self.bounds)

    def evaluate(self) -> float:
        
        return self.value

class SLayer:

    def __init__(self, nodes: list[SPoint]) -> None:
        self.nodes = nodes 

    def size(self) -> int:
        
        return len(self.nodes)
    
    def update_s_point_values(self, values: list[float]):
        
        for i in range(len(values)):
            self.nodes[i].update(values[i])

    def randomize(self) -> None:
        
        for node in self.nodes:
            node.randomize()

class Perceptron:
            
    def __init__(self, threshold: float = 1.0, output_value: float = 1.0) -> None:

        self.parent_nodes = None        
        self.weights = None

        self.threshold = threshold
        self.output_value = output_value

    def connect_parents(self, nodes: list[SPoint | Perceptron], weights: list[float]):        
        assert(len(nodes) == len(weights))
        
        self.parent_nodes = nodes
        self.weights = weights

    def update_weights(self, weights: list[float]):        
        assert(len(self.parent_nodes) == len(weights))
        self.weights = weights

    def aggregate_input_value(self):
        return sum([self.parent_nodes[x].evaluate() * self.weights[x] for x in range(len(self.parent_nodes))])

    def z(self):
        return self.aggregate_input_value() + self.threshold

    def activation_fn(self, z: float) -> float:
        return 1.0 if z > 0.0 else 0.0

    def evaluate(self) -> float:
        return self.output_value * self.activation_fn(self.z())

    def zero(self):
        self.update_weights([1.0 for x in self.weights])
        self.threshold = 1.0
        self.output_value = 1.0

class ALayer:
    
    def __init__(self, nodes: list[Perceptron]) -> None:
        self.nodes = nodes
        self.parent_layer: SLayer | ALayer | None = None
        
    def size(self):
        return len(self.nodes)

    def fully_connect_parent_layer(self, parent_layer: SLayer | ALayer, weights: list[float]):
        
        self.parent_layer = parent_layer

        for node in self.nodes:
            node.connect_parents(parent_layer.nodes, weights)

    def zero_nodes(self):
        for node in self.nodes:
            node.zero()
