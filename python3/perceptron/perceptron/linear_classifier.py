from __future__ import annotations

import logging

from random import uniform
from statistics import mean
from matplotlib import pyplot, lines

from perceptron.primitives import ALayer, Perceptron, SLayer, SPoint

class LinearClassifier:

    def __init__(self, x, bounds, weights, threshold, output_value) -> None:

        self.x_bounds = bounds[0] 
        self.x = SPoint(x[0], self.x_bounds)

        self.y_bounds = bounds[1]
        self.y = SPoint(x[1], self.y_bounds)

        self.s_layer = SLayer([self.x, self.y])

        perceptron = Perceptron(threshold, output_value)
        self.a_layer = ALayer([perceptron])
        self.a_layer.fully_connect_parent_layer(self.s_layer, weights)


    def generate_training_set(self, size: int):
        
        x_training = [(uniform(*self.x_bounds), uniform(*self.y_bounds)) for i in range(size)]

        inside = []
        outside = []

        for (x,y) in x_training:

            self.s_layer.update_s_point_values([x,y])
            final_output = self.a_layer.nodes[0].evaluate()

            if final_output > 0:
                inside.append((x,y))
            else:
                outside.append((x,y))

        # [inside_x, inside_y] = list(zip(*inside))
        # [outside_x, outside_y] = list(zip(*outside))

        return [ ([x,y], 1) for (x,y) in inside ] + [ ([x,y], 0) for (x,y) in outside ]

    def go(self, plotting_resolution: int, training_set_size: int):

        x_min = self.x_bounds[0]
        x_max = self.x_bounds[1]
        x_interval_size = x_max - x_min 
        x_step_size = x_interval_size / float(plotting_resolution)
        x_ = [x_min + (i * x_step_size) for i in range(plotting_resolution)]

        figure = pyplot.figure()
        subplot = figure.add_subplot(111)
            
        def line_graph_for_node(subplot, node: Perceptron, color = 'black'):
        
            a = node.weights[0]
            b = node.weights[1]
            c = node.threshold 

            y_ = [-1.0 * (a * x + c) / b for x in x_]

            return lines.Line2D(x_, y_, color=color)
            
        for node in self.a_layer.nodes:
            line_graph = line_graph_for_node(subplot, node)
            subplot.add_line(line_graph)

        def plot_sample(inside, outside):

            inside_x, inside_y = inside
            outside_x, outside_y = outside

            subplot.plot(inside_x, inside_y, 'x', color='black')
            subplot.plot(outside_x, outside_y, '.', color='black')

        subplot.grid(True, which='both')

        subplot.set_xlim(self.x_bounds)
        subplot.set_ylim(self.y_bounds)

        pyplot.show()

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


def test_linear_classifier():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            # logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

