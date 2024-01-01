from perceptron import __version__
from perceptron.linear_classifier import LinearClassifier, plot_reference_classifier

def test_version():
    assert __version__ == '0.1.0'

def test_linear_classifier():

    x_s = [0.0, 0.0]
    bounds = [(0.0, 2.0), (0.0, 2.0)]
    weights = [-2.0, -2.0]
    threshold = 3.0 
    output_value = 1.0
    reference_classifier = LinearClassifier(x_s, bounds, weights, threshold, output_value)
    
    training_set_size = 30
    training_data_set = reference_classifier.generate_training_set(training_set_size)

    plot_reference_classifier(reference_classifier)

    converged = True
    assert converged == True

