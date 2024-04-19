from random import uniform

from perceptron import __version__
from perceptron.chart import plot_classifier_with_training_data
from perceptron.networks import LinearClassifierNetwork

def test_version():
    assert __version__ == '0.0.1'

def test_linear_classifier_network():

    # produce training data set

    classifier_cardinality = 3
    l = 7
    x_min, x_max = -l, l
    y_min, y_max = -l, l
    input_bounds = [(x_min, x_max), (y_min, y_max)]

    classifier = LinearClassifierNetwork(classifier_cardinality, 2, input_bounds)

    training_set_size = 1000
    
    training_data_states = [
        (uniform(*classifier.input_bounds[0]), uniform(*classifier.input_bounds[1])) 
            for i in range(training_set_size)
    ] 

    training_data: list[tuple[tuple[float, float], int]] = [
        (state, classifier.classify_state(state))
        for state in training_data_states
    ]

    plot_classifier_with_training_data(classifier, training_data)

    assert(True)

# def test_training_of_linear_classifier():

#     classifier_cardinality = 1
#     dimension = 2
#     l = 10
#     x_min, x_max = -l, l
#     y_min, y_max = -l, l
#     input_bounds = [(x_min, x_max), (y_min, y_max)]

#     # randomly generate a reference classifier network
#     #
#     reference_classifer = LinearClassifierNetwork(classifier_cardinality, dimension, input_bounds)

#     # use the reference classifier to produce a set of training data

#     training_set_size = 1000
    
#     training_data_input_states = [
#         (uniform(*input_bounds.input_bounds[0]), uniform(*input_bounds.input_bounds[1])) 
#             for i in range(training_set_size)
#     ]
    
#     # x_, [w_.x_ > 0]_
#     #
#     training_data: list[tuple[tuple[float, float], int]] = [
#         (state, reference_classifer.classify_state(state))
#         for state in training_data_input_states
#     ]



#     # - generate a random classifier network for training
#     # - train the learning network using the reference data (run an epoch)
#     #   * during each epoch, graph
#     #     - the training data set so far
#     #     - the reference classifer
#     #     - the learning classifier
     




