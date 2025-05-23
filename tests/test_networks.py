from time import sleep

from perceptron import __version__
from perceptron.graphics.chart import new_axes, new_figure, plot_linear_classifier_network, plot_training_data
from perceptron.model.linear_classifier_network import LinearClassifierNetwork

import matplotlib
matplotlib.use("TkAgg")

from matplotlib import pyplot

from perceptron.train import random_alternating_training_data, train_linear_classifier_network

def test_version():
    assert __version__ == '0.0.1'

def test_generation_of_random_test_data_from_reference_classifier():

    classifier_cardinality = 1
    l = 7
    x_min, x_max = -l, l
    y_min, y_max = -l, l
    input_bounds = [(x_min, x_max), (y_min, y_max)]

    classifier = LinearClassifierNetwork(classifier_cardinality, 2, input_bounds)
    classifier.randomize()

    training_set_size = 50
    training_data = random_alternating_training_data(training_set_size, classifier)

    assert(len(training_data) == training_set_size)

def test_training_of_linear_classifier():

    classifier_cardinality = 1
    dimension = 2
    l = 10
    x_min, x_max = -l, l
    y_min, y_max = -l, l
    input_bounds = [(x_min, x_max), (y_min, y_max)]
    
    learning_rate = 0.25
    training_set_size = 1000
    epoch_count = 1

    # generate a (random) reference classifier network
    #
    reference_classifer = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    reference_classifer.randomize()

    # use the reference classifier to produce a set of training data
    #    
    training_data = random_alternating_training_data(training_set_size, reference_classifer)

    # generate a random classifier network for training
    #
    student_classifier = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    student_classifier.randomize()

    convergence_series: list[tuple[int, float]] = train_linear_classifier_network(
        student_classifier, 
        training_data,
        learning_rate=learning_rate,
        epochs=epoch_count,
        reference_classifier=reference_classifer) 

    convergence_figure = new_figure('convergence')
    
    convergence_bounds = [
        (0.0, float(training_set_size)),
        (0.0, 2.0),
    ]

    convergence_axes = new_axes(convergence_figure, convergence_bounds,scaled=False)

    n = [x[0] for x in convergence_series]
    distance = [x[1] for x in convergence_series]
    convergence_axes.plot(n, distance)#, '.', color='yellow')

    pyplot.get_current_fig_manager().window.wm_geometry("+800+0")
    pyplot.show(block=False)

    # ---------------------

    training_data_figure = new_figure('perceptrons (reference, student) with training data')
    axes = new_axes(training_data_figure, student_classifier.input_bounds)

    axes.set_xlim(input_bounds[0])
    axes.set_ylim(input_bounds[1])

    plot_training_data(axes, training_data)
    plot_linear_classifier_network(axes, reference_classifer, color='green')
    plot_linear_classifier_network(axes, student_classifier, color='purple')

    pyplot.show(block=False)

    pyplot.pause(20)
    pyplot.close()
