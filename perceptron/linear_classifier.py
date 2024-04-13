from __future__ import annotations

from random import uniform
from statistics import mean
from typing import List
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

    def run_training_set(self, training_set: list[tuple[list[float], int]]):

        self.a_layer.zero_nodes()

        learning_rate = 0.1

        error_ = []

        for (inputs, expected_output) in training_set:

            self.s_layer.update_s_point_values(inputs)
            actual_output = self.a_layer.nodes[0].evaluate()
            error = expected_output - actual_output

            error_.append(error)

            for node in self.a_layer.nodes:
                new_weights = [old_weight + (learning_rate * error) for old_weight in node.weights]
                node.update_weights(new_weights)
                node.threshold = node.threshold + (learning_rate * error)

        # graph error time series

        figure = pyplot.figure(f'error time series')
        subplot = figure.add_subplot(111)

        subplot.grid(True, which='both')

        subplot.set_xlim((0.0, len(error_)))
        subplot.set_ylim((min(error_), max(error_)))

        line_graph = lines.Line2D(range(len(error_)), error_, color='red')

        subplot.add_line(line_graph)

        pyplot.show()

def plot_linear_classifier(
        subplot, 
        classifier: LinearClassifier, 
        plotting_resolution: int = 100, 
        color = 'black'
    ):

    x_min = classifier.x_bounds[0]
    x_max = classifier.x_bounds[1]

    x_interval_size = x_max - x_min 
    x_step_size = x_interval_size / float(plotting_resolution)
    x_ = [x_min + (i * x_step_size) for i in range(plotting_resolution)]

    node = classifier.a_layer.nodes[0]

    a = node.weights[0]
    b = node.weights[1]
    c = node.threshold 

    y_ = [-1.0 * (a * x + c) / b for x in x_]

    line_graph = lines.Line2D(x_, y_, color=color)

    subplot.add_line(line_graph)

def plot_classifier_training_data(
        subplot, 
        training_data: list[tuple[list[float], float]]
    ):
    
    markers = ['x', '.', '*']

    categories = []

    for category_value in set([output_value for (_, output_value) in training_data]):
        categories.append((category_value, [xy for (xy, output_value) in training_data if output_value == category_value]))

    for i in range(len(categories)):
        (category_value, values) = categories[i]
        marker = markers[i]
        x, y = zip(*[(x_[0], x_[1]) for x_ in values])
        subplot.plot(x, y, marker, color='black')

def plot_reference_classifiers(
        classifiers: List[LinearClassifier], 
        training_data: list[tuple[list[float], int]]
    ):

    figure = pyplot.figure(f'reference classifier')
    subplot = figure.add_subplot(111)

    subplot.grid(True, which='both')
    
    # TODO determine common bounds of all classifiers

    subplot.set_xlim(classifiers[0].x_bounds)
    subplot.set_ylim(classifiers[0].y_bounds)

    for classifier in classifiers:
        plot_linear_classifier(subplot, classifier)
    
    plot_classifier_training_data(subplot, training_data)

    pyplot.show(block=False)
    pyplot.pause(3)
    pyplot.close()