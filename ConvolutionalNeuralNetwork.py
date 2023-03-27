import numpy as np

class node:

    def __init__(self,parent=[],children=[],id=0):
        self.parent = parent
        self.children = children
        self.id=id

    def start(self,image):
        prediction = self.prediction()


class layer:

    def __init__(self,parent=[],children=[]):
        self.parent = parent
        self.children = children
        self.nodes=[]

    def add_node(self,node):
        self._node.append(node)


class ConvolutionalNeuralNetwork:

    def __init__(self,output_number=2,name="no_name"):
        self.name = name
        self.output_number = output_number
        self.layers = []

    def add_layer(self,layer):
        self._layers.append(layer)
