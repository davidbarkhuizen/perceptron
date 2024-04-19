from matplotlib import pyplot, lines
from perceptron.networks import LinearClassifierNetwork

def plot(data_sets):

    figure = pyplot.figure()
    subplot = figure.add_subplot(111)

    x_all = []
    y_all = []

    for (label, x, y, color) in data_sets:

        line = lines.Line2D(x, y, color=color)
        subplot.add_line(line)

        x_all.extend(x)
        y_all.extend(y)

    subplot.set_xlim(min(x_all), max(x_all))
    subplot.set_ylim(min(y_all), max(y_all))

    pyplot.show(block=False)

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
