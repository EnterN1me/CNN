import numpy as np

from Layer import Layer, Dense


class Network:

    def __init__(self, input_size=-1):
        self.input_size = input_size
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def train(self, datas, iteration):
        for i_layer in range(len(self.layers)):
            for _ in range(iteration):
                for data, target in datas:
                    data = np.asarray(data)
                    self.layers[i_layer].learning((
                        self.prediction_layer(self.layers[i_layer - 1], data)
                        if i_layer > 0 else data), target
                    )

    def prediction_layer(self, layer, data):
        prediction = np.asarray(data)
        for current in self.layers:
            if current == layer:
                return current.prediction(prediction)
            prediction = current.prediction(prediction)

    def prediction(self, data):
        prediction = data
        for layer in self.layers:
            prediction = layer.prediction(prediction)
        return prediction
