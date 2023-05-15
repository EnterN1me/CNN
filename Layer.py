from Neuron import Neuron, Linear
from convolution import convolution, reLU, pooling, flatten, between_0_1
from abc import ABC
import numpy as np


class Layer(ABC):

    def prediction(self, data):
        pass

    def learning(self, data, target):
        pass


class Dense(Layer):

    def __init__(self, neuron_nbr, neuron_like):
        self.neuron_number = neuron_nbr
        if isinstance(neuron_like, Neuron):
            self.neurons = [
                Neuron(neuron_like.weights.shape[0], neuron_like.bias, neuron_like.learning_rate,
                       neuron_like.activation,
                       neuron_like.dactivation) for _ in range(neuron_nbr)]
        else:
            self.neurons = [Linear(neuron_like.weights.shape[0], neuron_like.bias, neuron_like.learning_rate) for _ in
                            range(neuron_nbr)]
        for neuron in self.neurons:
            neuron.set_random_weights()

    def prediction(self, data):
        return [neuron.prediction(data) for neuron in self.neurons]

    def learning(self, data, target):
        for neuron in self.neurons:
            neuron.learning(data, target)


class ConvolutionLayer(Layer):

    def __init__(self, kernels):
        self.kernels = kernels

    def prediction(self, image):
        image = np.asarray(image)
        return [np.asarray(
            reLU(convolution((sum(image) if len(image.shape) == 3 else image), kernel))
        ) for kernel in self.kernels]

    def learning(self, data, target): pass


class MaxPoolingLayer(Layer):

    def __init__(self, size):
        self.size = size

    def prediction(self, data):
        if len(np.asarray(data).shape) == 3:
            return [pooling(image, self.size) for image in data]
        return pooling(data, self.size)

    def learning(self, data, target): pass


class FlattenLayer(Layer):

    def __init__(self):
        pass

    def prediction(self, data):
        data = np.array(data)
        if len(data.shape) == 3:
            prediction = []
            for image in data:
                prediction.extend(flatten(between_0_1(image)))
            return prediction
        return flatten(data)

    def learning(self, data, target): pass
