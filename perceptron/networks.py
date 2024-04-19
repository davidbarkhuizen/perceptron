

from perceptron.primitives import AssociationLayer, AssociationNode, StateLayer
import random

class LinearClassifierNetwork:

    def __init__(self, 
        cardinality: int, 
        dimension: int,
        input_bounds: list[tuple[float, float]]
    ) -> None:

        self.cardinality = cardinality

        self.dimension = dimension

        assert(len(input_bounds) == dimension)
        self.input_bounds = input_bounds

        self.state_layer = StateLayer(dimension, input_bounds) 

        self.first_association_layer = AssociationLayer(
            nodes = [AssociationNode(
                parent_nodes=[node for node in self.state_layer.nodes],

                # NEXT: move this randomization out to a distinct function
                #
                parent_node_weights=[random.uniform(-2, 2) for node in self.state_layer.nodes],
                threshold=random.uniform(0, 3)

            ) for i in range(cardinality)],
            parent_layer=self.state_layer
        )

        # output a_layer consists of a 
        # - a single neuron
        #   * fully connected to all its parents
        #   * that activates when all its parent nodes are active
        # return sum([self.parent_nodes[i].evaluate() * self.parent_node_weights[i]

        self.final_association_layer = AssociationLayer(   
            nodes = [AssociationNode(
                parent_nodes=[node for node in self.first_association_layer.nodes],
                parent_node_weights=[1.0 for _ in self.first_association_layer.nodes],
                threshold=-(cardinality-1) 
            )],
        )

    def update_state_layer(self, x_: tuple[float]) -> None:
        self.state_layer.update_state(x_)

    def classify_state(self, state: tuple[float]) -> int:
        self.update_state_layer(state)
        return self.final_association_layer.nodes[0].evaluate()

    # def run_training_set(self, training_set: list[tuple[list[float], int]]):

    #     self.a_layer.zero_nodes()

    #     learning_rate = 0.1

    #     error_ = []

    #     for (inputs, expected_output) in training_set:

    #         self.s_layer.update_node_values(inputs)
    #         actual_output = self.a_layer.nodes[0].evaluate()
    #         error = expected_output - actual_output

    #         error_.append(error)

    #         for node in self.a_layer.nodes:
    #             new_weights = [old_weight + (learning_rate * error) for old_weight in node.parent_node_weights]
    #             node.update_weights(new_weights)
    #             node.threshold = node.threshold + (learning_rate * error)

    #     # graph error time series

    #     figure = pyplot.figure(f'error time series')
    #     subplot = figure.add_subplot(111)

    #     subplot.grid(True, which='both')

    #     subplot.set_xlim((0.0, len(error_)))
    #     subplot.set_ylim((min(error_), max(error_)))

    #     line_graph = lines.Line2D(range(len(error_)), error_, color='red')

    #     subplot.add_line(line_graph)

    #     pyplot.show()
