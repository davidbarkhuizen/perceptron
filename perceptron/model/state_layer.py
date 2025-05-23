from perceptron.model.state_node import StateNode


class StateLayer:
    '''
    sense layer'''

    def __init__(self, dimension: int, bounds: list[tuple[float, float]]) -> None:
        assert(len(bounds) == dimension)
        
        self.dimension = dimension
        self.nodes = [StateNode(bounds[i]) for i in range(dimension)]
    
    def update_state(self, x_: tuple[float]) -> None:
        assert(len(x_) == self.dimension)

        for i in range(self.dimension):
            self.nodes[i].update_value(x_[i])

    def randomize(self) -> None:
        
        for node in self.nodes:
            node.randomize()