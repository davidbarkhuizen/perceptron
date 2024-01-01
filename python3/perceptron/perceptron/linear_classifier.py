from __future__ import annotations

from random import uniform
from statistics import mean
from matplotlib import pyplot, lines

import logging

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
    
    def randomize(self) -> None:
        
        for node in self.nodes:
            node.randomize()

class Perceptron:
            
    def __init__(self, threshold: float, output_value: float) -> None:

        self.parent_nodes = None        
        self.w = None

        self.threshold = threshold
        self.output_value = output_value

    def connect_parents(self, nodes: list[SPoint | Perceptron], weights: list[float]):        
        assert(len(nodes) == len(weights))
        
        self.parent_nodes = nodes
        self.w = weights

    def aggregate_input_value(self):
        return sum([self.parent_nodes[x].evaluate() * self.w[x] for x in range(len(self.parent_nodes))])

    def z(self):
        return self.aggregate_input_value() + self.threshold

    def activation_fn(z: float) -> float:
        return 1.0 if z > 0.0 else 0.0

    def evaluate(self) -> float:
        return self.output_value * self.activation_fn(self.z())

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

class SimpleLinearClassifier:

    def __init__(self) -> None:

        self.x_bounds = (0.0, 2.0)
        x = SPoint(mean(self.x_bounds), self.x_bounds)

        self.y_bounds = (0.0, 2.0)
        y = SPoint(mean(self.y_bounds), self.y_bounds)

        self.s_layer = SLayer([x, y])

        # ax, by
        s_node_weights = [-2.0, -2.0]

        perceptron = Perceptron(threshold=3.0, output_value=1.0)
        
        self.a_layer = ALayer([perceptron])
        self.a_layer.fully_connect_parent_layer(self.s_layer, s_node_weights)

    def graph(self):

        figure = pyplot.figure()
        subplot = figure.add_subplot(111)

        sample_size = 30

        x_min = self.x_bounds[0]
        x_max = self.x_bounds[1]
        x_interval_size = x_max - x_min 
        x_step_size = x_interval_size / float(sample_size)
        
        logging.info(f'x_min {x_min}, x_max {x_max}, x_interval_size {x_interval_size}, x_step_size {x_step_size}')
    
        x_ = [x_min + (i * x_step_size) for i in range(sample_size)]
        
        def graph_a_node(node: Perceptron, color = 'black'):
        
            a = node.w[0]
            b = node.w[1]
            c = node.threshold 

            y_ = [-1.0 * (a * x + c) / b for x in x_]

            line = lines.Line2D(x_, y_, color=color)
            subplot.add_line(line)

        for node in self.a_layer.nodes:
            graph_a_node(node)

        subplot.grid(True, which='both')

        subplot.set_xlim(self.x_bounds)
        subplot.set_ylim(self.y_bounds)

        pyplot.show()

def entrypoint():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            # logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    classifier = SimpleLinearClassifier()
    classifier.graph()

entrypoint()