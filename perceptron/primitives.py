from __future__ import annotations
from random import uniform

class StateNode:
    '''
    sense point
    '''

    def __init__(self, bounds: tuple[float, float], value: float = 0.0) -> None:

        self.value = value
        self.bounds = bounds
        
    def update_value(self, value: float) -> None:
        
        self.value = value

    def randomize(self) -> None:
        
        self.value = uniform(*self.bounds)

    def activation_fn(self) -> float:
        
        return self.value

class StateLayer:
    '''
    sense layer'''

    def __init__(self, dimension: int, bounds: list[tuple[float, float]]) -> None:
        assert(len(bounds) == dimension)
        
        self.dimension = dimension
        self.nodes = [StateNode(bounds[i]) for i in range(dimension)]
    
    def update_state(self, x_: tuple[float]) -> None:
        
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
        self.parent_node_weights = parent_node_weights if parent_node_weights is not None else []

    def update_parent_weights(self, weights: list[float]) -> None:                
        assert(len(weights) == len(self.parent_nodes))
        self.parent_node_weights.clear()
        self.parent_node_weights.extend(weights)

    def z(self) -> float:

        aggregate_input_value = sum([
            self.parent_nodes[i].activation_fn() * self.parent_node_weights[i] 
                for i in range(len(self.parent_nodes))
        ])

        return aggregate_input_value + self.threshold

    def activation_fn(self) -> int:
        return 1 if self.z() > 0.0 else 0

class AssociationLayer:
    '''
    association layers
    '''
    
    def __init__(self, 
        nodes: list[AssociationNode], 
        parent_layer: StateLayer | AssociationLayer | None = None
    ) -> None:
        
        self.nodes = nodes
        self.parent_layer = parent_layer

    def size(self) -> int:
        return len(self.nodes)

    def randomize(self):
        for node in self.nodes:
            node.update_parent_weights()

