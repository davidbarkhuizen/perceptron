description = 'perceptron'
print(description)

from graphx import plot

data_sets = [
    tuple(('alpha', [10,24,23,23,3], [12,2,3,4,2], 'yellow')),
    tuple(('beta', [2,4,8,16,32], [12,2,3,4,2], 'purple'))
]

plot(data_sets)




