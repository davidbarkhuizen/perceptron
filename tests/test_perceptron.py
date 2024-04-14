from random import uniform

from perceptron import __version__
from perceptron.linear_classifier import new_linear_classifer, plot_reference_classifiers

def test_version():
    assert __version__ == '0.1.0'

def test_linear_classifier():

    for i in range(5):

        x_ = [0.0, 0.0]

        x_min = -5.0
        x_max = 5.0

        y_min = -5.0
        y_max = 5.0

        bounds = [(x_min, x_max), (y_min, y_max)]
        
        classifier_count = 3
        classifiers = []

        training_set_size = 1000
        training_data_set = []

        for i in range(classifier_count): 
            classifiers.append(new_linear_classifer(x_, bounds))
                
        classifier = classifiers[0]
        x_training = [(uniform(*classifier.x_bounds), uniform(*classifier.y_bounds)) for i in range(training_set_size)]

        for (x,y) in x_training:

            classifier_outputs = []

            for classifier in classifiers:

                classifier.s_layer.update_s_point_values([x,y])
                classifier_outputs.append(classifier.a_layer.nodes[0].evaluate())

            intersection = 1 if sum(classifier_outputs) == classifier_count else 0

            training_data_set.append(([x,y], intersection))

        plot_reference_classifiers(classifiers, training_data_set)
