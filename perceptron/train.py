from random import shuffle, uniform
from perceptron.networks import LinearClassifierNetwork

def random_alternating_training_data(
        size: int, 
        classifier: LinearClassifierNetwork
    ) -> list[tuple[tuple[float, float], int]]:

    states = { 0: [], 1: []}

    k = size // 2

    while len(states[0]) < k or len(states[1]) < k:

        input = (uniform(*classifier.input_bounds[0]), uniform(*classifier.input_bounds[1]))
        state = classifier.classify_state(input)

        if len(states[state]) < k:
            states[state].append((input, state))
    
    mixed = states[0] + states[1]
    shuffle(mixed)
    return mixed

def train_linear_classifier_network(
    classifier: LinearClassifierNetwork,
    training_data: list[tuple[tuple[float, float], int]],
    learning_rate: float = 0.25,
    epochs: int = 1,
    reference_classifier: LinearClassifierNetwork = None
):

    iterations = 0
    convergence = []

    convergence.append((iterations, reference_classifier.distance(classifier)))

    for _ in range(epochs):        
        for datum in training_data:
            (x_, reference_category) = datum
            classifier.teach(learning_rate, x_, reference_category)
            iterations += 1

            if reference_classifier is not None:
                convergence.append((iterations, reference_classifier.distance(classifier)))

    return convergence