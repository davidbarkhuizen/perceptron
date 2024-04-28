from time import sleep

from perceptron import __version__
from perceptron.chart import new_axes, new_figure, plot_linear_classifier_network, plot_training_data
from perceptron.networks import LinearClassifierNetwork

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
    epochs = 1

    # generate a (random) reference classifier network
    #
    reference_classifer = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    reference_classifer.randomize()

    # use the reference classifier to produce a set of training data
    #    
    training_data = random_alternating_training_data(training_set_size, reference_classifer)

    # generate a random classifier network for training
    #
    classifier = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    classifier.randomize()

    convergence = train_linear_classifier_network(
        classifier, 
        training_data,
        learning_rate=0.25,
        epochs=1,
        reference_classifier=reference_classifer) 

    figure = new_figure('perceptron convergence')
    
    convergence_bounds = [
        (min(convergence, key=lambda x: x[0])[0], max(convergence, key=lambda x: x[0])[0]),
        (min(convergence, key=lambda x: x[1])[1], max(convergence, key=lambda x: x[1])[1]),
    ]

    print(convergence_bounds)

    convergence_axes = new_axes(figure, convergence_bounds,scaled=False)

    n = [x[0] for x in convergence]
    distance = [x[1] for x in convergence]
    convergence_axes.plot(n, distance, '.', color='yellow')

    # ---------------------

    figure = new_figure('learning perceptron')
    axes = new_axes(figure, classifier.input_bounds)

    axes.set_xlim(input_bounds[0])
    axes.set_ylim(input_bounds[1])

    plot_training_data(axes, training_data)
    plot_linear_classifier_network(axes, reference_classifer, color='green')
    plot_linear_classifier_network(axes, classifier, color='purple')

    pyplot.show(block=False)
    pyplot.pause(20)
    pyplot.close()
