import time

import matplotlib.pyplot as plt

from FeatureExtraction import FeatureExtraction
from convolution import convolution, reLU, pooling, Kernel_3x3, between_0_1, flatten
from NeuralNetwork import Neuron, Lineaire, tanh, dtanh
import cv2
import numpy as np
from Data import Dataset

# ====================sobel_horizontal===================#
sobel_horizontal = FeatureExtraction()  # 48x48

sobel_horizontal.add_layer(convolution, Kernel_3x3["sobel_horizontal"])  # 46x46
sobel_horizontal.add_layer(reLU)

sobel_horizontal.add_layer(pooling, 2)  # 23x23

sobel_horizontal.add_layer(convolution, Kernel_3x3["sobel_horizontal"])  # 21x21
sobel_horizontal.add_layer(reLU)

sobel_horizontal.add_layer(pooling, 2)  # 11x11

sobel_horizontal.add_layer(convolution, Kernel_3x3["sobel_horizontal"])  # 9x9
sobel_horizontal.add_layer(reLU)

sobel_horizontal.add_layer(between_0_1)
sobel_horizontal.add_layer(flatten)

# ====================sobel_vertical===================#
sobel_vertical = FeatureExtraction()  # 48x48

sobel_vertical.add_layer(convolution, Kernel_3x3["sobel_vertical"])  # 46x46
sobel_vertical.add_layer(reLU)

sobel_vertical.add_layer(pooling, 2)  # 23x23

sobel_vertical.add_layer(convolution, Kernel_3x3["sobel_vertical"])  # 21x21
sobel_vertical.add_layer(reLU)

sobel_vertical.add_layer(pooling, 2)  # 11x11

sobel_vertical.add_layer(convolution, Kernel_3x3["sobel_vertical"])  # 9x9
sobel_vertical.add_layer(reLU)

sobel_vertical.add_layer(between_0_1)
sobel_vertical.add_layer(flatten)

# ====================contour===================#
contour = FeatureExtraction()  # 48x48

contour.add_layer(convolution, Kernel_3x3["contour"])  # 46x46
contour.add_layer(reLU)

contour.add_layer(pooling, 2)  # 23x23

contour.add_layer(convolution, Kernel_3x3["contour"])  # 21x21
contour.add_layer(reLU)

contour.add_layer(pooling, 2)  # 11x11

contour.add_layer(convolution, Kernel_3x3["contour"])  # 9x9
contour.add_layer(reLU)

contour.add_layer(between_0_1)
contour.add_layer(flatten)

# ====================blur===================#
blur = FeatureExtraction()  # 48x48

blur.add_layer(convolution, Kernel_3x3["blur"])  # 46x46
blur.add_layer(reLU)

blur.add_layer(pooling, 2)  # 23x23

blur.add_layer(convolution, Kernel_3x3["blur"])  # 21x21
blur.add_layer(reLU)

blur.add_layer(pooling, 2)  # 11x11

blur.add_layer(convolution, Kernel_3x3["blur"])  # 9x9
blur.add_layer(reLU)

blur.add_layer(between_0_1)
blur.add_layer(flatten)


# ======= #

hidden_neuron = [Neuron(21 * 21, activation=tanh, dactivation=dtanh) for _ in range(4)]
for neuron in hidden_neuron:
    neuron.set_random_weights(2)
output_neuron = hidden_neuron(4, activation=tanh, dactivation=dtanh)
output_neuron.set_random_weights(2)


# pizza = 1 // avion = -1
def training():
    # neuron_hor = Neuron(np.zeros(21 * 21), learning_rate=0.05, activation=tanh, dactivation=dtanh,
    #                    possible_value=[-1, 1])

    neuron = Lineaire(21 * 21)
    neuronBis = Neuron(21 * 21, activation=tanh, dactivation=dtanh)

    # DATASET
    train_nbr = 100

    dt = Dataset("Dataset\\96")
    pizza = dt.get_new_dataset(train_nbr, "pizza", "jpg", 1)
    pizza = dt.add_feature_to_dataset(pizza, sobel_horizontal)
    avion = dt.get_new_dataset(train_nbr, "avion", "jpg", -1)
    avion = dt.add_feature_to_dataset(avion, sobel_horizontal)

    # TRAIN

    for _ in range(100):

        for i in range(train_nbr):
            prediction = [neuron.prediction(pizza[i][0]) for neuron in hidden_neuron]
            for neuron in hidden_neuron:
                neuron.learning(pizza[i][0], pizza[i][1])
            output_neuron.learning(prediction, pizza[i][1])
            prediction = [neuron.prediction(avion[i][0]) for neuron in hidden_neuron]
            for neuron in hidden_neuron:
                neuron.learning(avion[i][0], avion[i][1])
            output_neuron.learning(prediction, avion[i][1])

    # TEST DATASET

    test_nbr = 24

    pizza = dt.get_new_dataset(test_nbr, "pizza", "jpg", 1, 101)
    pizza = dt.add_feature_to_dataset(pizza, sobel_horizontal)
    avion = dt.get_new_dataset(test_nbr, "avion", "jpg", -1, 101)
    avion = dt.add_feature_to_dataset(avion, sobel_horizontal)

    res = 0
    for i in pizza:
        prediction = [neuron.prediction(i[0]) for neuron in hidden_neuron]
        res += (1 if output_neuron.prediction(prediction) > 0 else 0)

    for i in avion:
        prediction = [neuron.prediction(i[0]) for neuron in hidden_neuron]
        res += (1 if output_neuron.prediction(prediction) < 0 else 0)

    print(res * 100 / (2*test_nbr))


training()


def get_value(image):
    image = contour.execute(image)
    prediction = [neuron.prediction(image) for neuron in hidden_neuron]
    print((f"{n*100}% pizza" if (n := output_neuron.prediction(prediction) > 0) else f"{abs(n)*100}avion"))