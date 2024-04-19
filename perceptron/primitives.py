from __future__ import annotations
from random import uniform

class StateNode:
    '''
    sense point
    '''

    def __init__(self, bounds: tuple[float, float], value: float = 0.0) -> None:

        self.__value = value
        self.bounds = bounds

    def randomize(self) -> None:
        self.__value = uniform(*self.bounds)

    def value(self) -> float:
        return self.__value

    def update_value(self, value: float) -> None:
        self.__value = value


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

class AssociationNode:
            
    def __init__(self, 
        threshold: float = 0.0,
        parent_nodes: list[StateNode | AssociationNode] = None, 
        parent_node_weights: list[float] = None
    ) -> None:

        self.threshold = threshold

        self.parent_nodes = parent_nodes if parent_nodes is not None else []        
        self.parent_node_weights = parent_node_weights if parent_node_weights is not None else [
            1.0 for _ in self.parent_nodes
        ]

    def update_parent_weights(self, weights: list[float]) -> None:                
        assert(len(weights) == len(self.parent_nodes))
        self.parent_node_weights = weights

    def z(self) -> float:

        aggregate_input_value = sum([
            self.parent_nodes[i].value() * self.parent_node_weights[i]
                for i in range(len(self.parent_nodes))
        ])

        return aggregate_input_value + self.threshold

    def value(self) -> int:
        return 1 if self.z() > 0.0 else 0

    def teach(self, target_state: int):

        print(f'state: {target_state}')

        current_state = self.value()
        correctly_categorised = current_state == target_state
        if correctly_categorised:
            return

        if target_state == 1 and current_state == 0:
            d = 1.0
        elif target_state == 0 and current_state == 1:
            d = -1.0
        else:
            raise('wtf')

        n = 0.2

        w_p0 = self.threshold + n * d * 1.0
        w_p1 = self.parent_node_weights[0] + n * d * self.parent_nodes[0].value()
        w_p2 = self.parent_node_weights[1] + n * d * self.parent_nodes[1].value()

        self.threshold = w_p0
        self.update_parent_weights([w_p1, w_p2])

class AssociationLayer:
    '''
    association layers
    '''
    
    def __init__(self, size, parent_layer: StateLayer | AssociationLayer) -> None:
        
        self.size = size
        self.parent_layer = parent_layer

        self.nodes = [
            AssociationNode(parent_nodes=self.parent_layer.nodes) 
                for i in range(size)
        ]
