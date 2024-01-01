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
        assert(len(self.nodes) == len(weights))
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
        self.update_weights([0.0 for x in self.weights])
        self.threshold = 1
        self.output_value = 1

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

class SimpleLinearClassifier:

    def __init__(self) -> None:

        self.x_bounds = (0.0, 2.0)
        self.x = SPoint(mean(self.x_bounds), self.x_bounds)

        self.y_bounds = (0.0, 2.0)
        self.y = SPoint(mean(self.y_bounds), self.y_bounds)

        self.s_layer = SLayer([self.x, self.y])

        # ax, by
        s_node_weights = [-2.0, -2.0]

        perceptron = Perceptron(threshold=3.0, output_value=1.0)
        
        self.a_layer = ALayer([perceptron])
        self.a_layer.fully_connect_parent_layer(self.s_layer, s_node_weights)

    def go(self):

        figure = pyplot.figure()
        subplot = figure.add_subplot(111)

        sample_size = 30

        x_min = self.x_bounds[0]
        x_max = self.x_bounds[1]
        x_interval_size = x_max - x_min 
        x_step_size = x_interval_size / float(sample_size)
        
        logging.info(f'x_min {x_min}, x_max {x_max}, x_interval_size {x_interval_size}, x_step_size {x_step_size}')
    
        x_ = [x_min + (i * x_step_size) for i in range(sample_size)]
        
        def graph_a_node(subplot, node: Perceptron, color = 'black'):
        
            a = node.weights[0]
            b = node.weights[1]
            c = node.threshold 

            y_ = [-1.0 * (a * x + c) / b for x in x_]

            line = lines.Line2D(x_, y_, color=color)
            subplot.add_line(line)

        for node in self.a_layer.nodes:
            graph_a_node(subplot, node)

        random_sample = [(uniform(*self.x_bounds), uniform(*self.y_bounds)) for i in range(sample_size)]

        inside = []
        outside = []

        for (x,y) in random_sample:

            self.s_layer.update_s_point_values([x,y])
            final_output = self.a_layer.nodes[0].evaluate()

            if final_output > 0:
                inside.append((x,y))
            else:
                outside.append((x,y))

        [inside_x, inside_y] = list(zip(*inside))
        subplot.plot(inside_x, inside_y, 'x', color='black')

        [outside_x, outside_y] = list(zip(*outside))
        subplot.plot(outside_x, outside_y, '.', color='black')

        subplot.grid(True, which='both')

        subplot.set_xlim(self.x_bounds)
        subplot.set_ylim(self.y_bounds)

        pyplot.show()

        training_set = [ ([x,y], 1) for (x,y) in inside ] + [ ([x,y], 0) for (x,y) in outside ]

        self.a_layer.zero_nodes()

        learning_rate = 0.1

        for (inputs, expected_output) in training_set:

            self.s_layer.update_s_point_values(inputs)
            actual_output = self.a_layer.nodes[0].evaluate()
            error = expected_output - actual_output

            for node in self.a_layer.nodes:
                new_weights = [old_weight + (learning_rate * error) for old_weight in node.weights]
                node.update_weights(new_weights)

            # graph current state of node

def entrypoint():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            # logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    reference = SimpleLinearClassifier()
    reference.go()

entrypoint()