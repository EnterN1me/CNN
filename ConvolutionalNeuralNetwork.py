import numpy as np
from FeatureExtraction import FeatureExtraction
from convolution import between_0_1,flatten

class Node:

    def __init__(self, parent=[], children=[], id=0):
        self.parent = parent
        self.children = children
        self.id=id

    def start(self,image):
        prediction = self.prediction()


class Layer:

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
        self.first_feature_node = []
        self.last_feature_node = []

    def first_feature_execute(self, dataset:list[list[np.ndarray,float]]):
        return self.feature_execute(self.first_feature_node,dataset)

    def feature_execute(self,feature_liste, dataset:list[list[np.ndarray,float]]):
        dataset = np.asarray(dataset)
        new_dataset = []

        for image in dataset:
            for node in feature_liste:
                image[0] = node.execute(image[0])
                new_dataset.append([image[0], image[1]])  # image 0 = image, image 1 = valeur associe

        return new_dataset

    def last_feature_execute(self,dataset:list[list[np.ndarray,float]]):
        dataset = self.feature_execute(self.last_feature_node,dataset)
        activation = FeatureExtraction
        activation.add_layer(between_0_1)
        activation.add_layer(flatten)
        
        dataset = self.feature_execute([activation],dataset)

    def add_layer(self, layer):
        self.features_layers_layers.append(layer)
