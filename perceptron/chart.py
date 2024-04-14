from matplotlib import pyplot, lines

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