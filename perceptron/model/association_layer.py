from __future__ import annotations
from perceptron.model.association_node import AssociationNode
from perceptron.model.state_layer import StateLayer


class AssociationLayer:
    '''
    association layers
    '''
    
    def __init__(self, 
        size: int, 
        input_layer: StateLayer | AssociationLayer
    ) -> None:
        
        self.size = size
        self.input_layer = input_layer

        self.nodes = [
            AssociationNode(inpute_nodes=self.input_layer.nodes) 
                for i in range(size)
        ]