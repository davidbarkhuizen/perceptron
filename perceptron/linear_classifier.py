from __future__ import annotations

from random import uniform
import random
from statistics import mean
from typing import List
from matplotlib import pyplot, lines

from perceptron.networks import LinearClassifierNetwork
from perceptron.primitives import AssociationLayer, AssociationNode, StateLayer, StateNode

    # def run_training_set(self, training_set: list[tuple[list[float], int]]):

    #     self.a_layer.zero_nodes()

    #     learning_rate = 0.1

    #     error_ = []

    #     for (inputs, expected_output) in training_set:

    #         self.s_layer.update_node_values(inputs)
    #         actual_output = self.a_layer.nodes[0].evaluate()
    #         error = expected_output - actual_output

    #         error_.append(error)

    #         for node in self.a_layer.nodes:
    #             new_weights = [old_weight + (learning_rate * error) for old_weight in node.parent_node_weights]
    #             node.update_weights(new_weights)
    #             node.threshold = node.threshold + (learning_rate * error)

    #     # graph error time series

    #     figure = pyplot.figure(f'error time series')
    #     subplot = figure.add_subplot(111)

    #     subplot.grid(True, which='both')

    #     subplot.set_xlim((0.0, len(error_)))
    #     subplot.set_ylim((min(error_), max(error_)))

    #     line_graph = lines.Line2D(range(len(error_)), error_, color='red')

    #     subplot.add_line(line_graph)

    #     pyplot.show()

def plot_linear_classifier_network(
        subplot, 
        classifier: LinearClassifierNetwork, 
        plotting_resolution: int = 100, 
        color = 'black'
    ):

    x_min = classifier.input_bounds[0][0]
    x_max = classifier.input_bounds[0][1]

    x_interval_size = x_max - x_min 
    x_step_size = x_interval_size / float(plotting_resolution)
    x_ = [x_min + (i * x_step_size) for i in range(plotting_resolution)]

    for node in classifier.first_association_layer.nodes:

        a = node.parent_node_weights[0]
        b = node.parent_node_weights[1]
        c = node.threshold 

        y_ = [-1.0 * (a * x + c) / b for x in x_]

        line_graph = lines.Line2D(x_, y_, color=color)

        subplot.add_line(line_graph)

def plot_training_data(
        subplot, 
        training_data: list[tuple[tuple[float, float], int]]
    ):
    
    markers = ['.', 'x']

    categories = []

    for category_value in set([output_value for (_, output_value) in training_data]):
        categories.append((category_value, [xy for (xy, output_value) in training_data if output_value == category_value]))

    for i in range(len(categories)):
        (category_value, values) = categories[i]
        marker = markers[i]
        x, y = zip(*[(x_[0], x_[1]) for x_ in values])
        subplot.plot(x, y, marker, color='black')

def plot_classifier_with_training_data(
        classifier: LinearClassifierNetwork, 
        training_data: list[tuple[tuple[float, float], int]]
    ):

    figure = pyplot.figure(f'reference classifier')
    subplot = figure.add_subplot(111)

    subplot.grid(True, which='both')
    
    subplot.set_xlim(classifier.input_bounds[0])
    subplot.set_ylim(classifier.input_bounds[1])

    plot_linear_classifier_network(subplot, classifier)
    
    plot_training_data(subplot, training_data)

    pyplot.show(block=False)
    pyplot.pause(5)
    pyplot.close()