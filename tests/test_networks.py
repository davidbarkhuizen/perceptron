from random import uniform
from time import sleep

from perceptron import __version__
from perceptron.chart import new_axes, new_figure, plot_linear_classifier_network, plot_training_data
from perceptron.networks import LinearClassifierNetwork

from matplotlib import pyplot

def test_version():
    assert __version__ == '0.0.1'

def random_training_data(size: int, classifier: LinearClassifierNetwork) -> list[tuple[tuple[float, float], int]]:

    training_data_states = [
        (uniform(*classifier.input_bounds[0]), uniform(*classifier.input_bounds[1])) 
            for i in range(size)
    ] 

    return [
        (state, classifier.classify_state(state))
        for state in training_data_states
    ]

def test_generation_of_random_test_data_from_reference_classifier():

    classifier_cardinality = 1
    l = 7
    x_min, x_max = -l, l
    y_min, y_max = -l, l
    input_bounds = [(x_min, x_max), (y_min, y_max)]

    classifier = LinearClassifierNetwork(classifier_cardinality, 2, input_bounds)
    classifier.randomize()

    training_set_size = 67

    assert(len(random_training_data(training_set_size, classifier)) == training_set_size)

def test_training_of_linear_classifier():

    classifier_cardinality = 1
    dimension = 2
    l = 10
    x_min, x_max = -l, l
    y_min, y_max = -l, l
    input_bounds = [(x_min, x_max), (y_min, y_max)]
    
    learning_rate = 0.2
    training_set_size = 100
    epochs = 3

    # generate a (random) reference classifier network
    #
    reference_classifer = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    reference_classifer.randomize()

    # use the reference classifier to produce a set of training data
    #
    training_data_input_states = [
        (uniform(*input_bounds[0]), uniform(*input_bounds[1])) 
            for i in range(training_set_size)
    ]
    
    # x_, [w_.x_ > 0]_
    #
    training_data: list[tuple[tuple[float, float], int]] = [
        (state, reference_classifer.classify_state(state))
        for state in training_data_input_states
    ]

    # generate a random classifier network for training
    #
    classifier = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)
    classifier.randomize()

    # train the naive network using the reference data (run an epoch)
    #
    for _ in range(epochs):        
        for datum in training_data:
            (x_, reference_category) = datum
            classifier.teach(learning_rate, x_, reference_category)

    figure = new_figure('taught perceptron')
    axes = new_axes(figure, classifier.input_bounds)
    pyplot.show(block=False)

    axes.clear()
    axes.set_xlim(input_bounds[0])
    axes.set_ylim(input_bounds[1])

    plot_training_data(axes, training_data)
    plot_linear_classifier_network(axes, reference_classifer, color='green')
    plot_linear_classifier_network(axes, classifier, color='purple')
    
    pyplot.pause(5)
    pyplot.close()


