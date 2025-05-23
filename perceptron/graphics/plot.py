from matplotlib import pyplot, lines
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from perceptron.model.linear_classifier_network import LinearClassifierNetwork

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
        axes: Axes, 
        classifier: LinearClassifierNetwork, 
        plotting_resolution: int = 100, 
        color = 'purple'
    ):

    x_min = classifier.input_bounds[0][0]
    x_max = classifier.input_bounds[0][1]

    x_interval_size = x_max - x_min 
    x_step_size = x_interval_size / float(plotting_resolution)
    x_ = [x_min + (i * x_step_size) for i in range(plotting_resolution)]

    for node in classifier.hidden_layer.nodes:

        a = node.input_node_weights[0]
        b = node.input_node_weights[1]
        c = node.threshold 

        y_ = [-1.0 * (a * x + c) / b for x in x_]

        line_graph = lines.Line2D(x_, y_, color=color)

        axes.add_line(line_graph)

def plot_training_data(
        axes: Axes, 
        training_data: list[tuple[tuple[float, float], int]]
    ):
    
    markers = ['.', 'x']
    colors = ['blue', 'yellow']

    categories = []

    for category_value in set([output_value for (_, output_value) in training_data]):
        categories.append((category_value, [xy for (xy, output_value) in training_data if output_value == category_value]))

    for i in range(len(categories)):
        (category_value, values) = categories[i]
        marker = markers[i]
        color = colors[i]
        x, y = zip(*[(x_[0], x_[1]) for x_ in values])
        axes.plot(x, y, marker, color=color)

def new_figure(label: str) -> Figure:
    figure = pyplot.figure(label)
    figure.patch.set_facecolor('xkcd:black')
    return figure

def new_axes(figure: Figure, bounds: list[tuple[float, float]]) -> Axes:

    axes = figure.add_subplot(111)
    axes.set_facecolor('xkcd:black')

    axes.grid(True, which='both')
    
    axes.set_xlim(bounds[0])
    axes.set_ylim(bounds[1])

    axes.set_aspect('equal', adjustable='box')

    axes.spines['bottom'].set_color('white')
    axes.spines['top'].set_color('white')
    axes.spines['left'].set_color('white')
    axes.spines['right'].set_color('white')

    axes.xaxis.label.set_color('white')
    axes.tick_params(axis='x', colors='white')
    axes.tick_params(axis='y', colors='white')



    return axes