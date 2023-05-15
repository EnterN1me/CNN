import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

from time import time

import Neuron
import convolution


def test_linear():
    # basic and calculus # =========================== #
    init_weight_number = 1
    init_bias = 1
    neuron = Neuron.Linear(init_weight_number, init_bias, 0.01)
    assert neuron.prediction(2) == 1, "prediction error : " + str(neuron.prediction(2)) + str(neuron.weights) + str(
        neuron.bias)
    neuron.learning(2, 4)
    assert neuron.weights == np.zeros(init_weight_number) - np.multiply(neuron.learning_rate,
                                                                        (2 * (0 * 2 + 1 - 4)))  # 1.2
    assert neuron.bias == init_bias - neuron.learning_rate * (0 * 2 + 1 - 4)
    assert neuron.prediction(2) > 1
    del neuron

    # random weights # =========================== #
    neuron = Neuron.Linear()
    neuron.set_random_weights()
    assert neuron.weights != [0], "random initialisation error " + str(neuron.weights)


def test_convolution(filename="test_image.jpg"):
    image = cv2.imread(filename, 0)
    image = cv2.resize(image, (96,96))
    image = np.asarray(image)

    start = time()

    #image = convolution.convolution(image, convolution.Kernel_3x3["blur"])
    convolution.reLU(image)

    plt.figure("1 convolution / conv+ (pool+conv)*2")
    plt.imshow(image)

    print(time() - start, "s", file=sys.stderr)
    plt.show()


if __name__ == "__main__":
    test_linear()
    test_convolution()
