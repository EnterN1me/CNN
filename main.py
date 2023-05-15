import os
import sys
import time

from matplotlib import pyplot as plt

import Data
from convolution import convolution, reLU, pooling, Kernel_3x3, between_0_1, flatten
from Neuron import Neuron, Linear, tanh, dtanh
import random
from Network import Network
from Layer import ConvolutionLayer, MaxPoolingLayer, Dense, FlattenLayer
import cv2
import numpy as np
from tqdm import tqdm

# ======= #


ConvolutionPart = [
    ConvolutionLayer(
        [Kernel_3x3["sobel_horizontal"], Kernel_3x3["sobel_vertical"]]),
    MaxPoolingLayer(2),
    ConvolutionLayer(
        [Kernel_3x3["sobel_horizontal"], Kernel_3x3["sobel_vertical"]]),
    MaxPoolingLayer(2),
    ConvolutionLayer(
        [Kernel_3x3["sobel_horizontal"], Kernel_3x3["sobel_vertical"]]),
    FlattenLayer(),
]


def training(train, test, iteration=15, nbr_neuron=6):
    data = Data.dataset_tolist()
    sample_size = len(data)
    random.shuffle(data)

    for i in tqdm(range(sample_size)):
        for layer in ConvolutionPart:
            data[i][0] = layer.prediction(data[i][0])

    model = Network()
    model.add(Dense(nbr_neuron, Neuron(21 * 21 * 2, learning_rate=0.01, activation=tanh, dactivation=dtanh)))
    model.add(Dense(1, Neuron(nbr_neuron, learning_rate=0.01, activation=tanh, dactivation=dtanh)))

    # TRAIN

    data_train = data[:train]
    model.train(data_train, iteration)

    # TEST DATASET

    data_test = data[train:train + test]
    res = 0
    for data_temp, target in data_test:
        res += (1 if abs(model.prediction(data_temp)[0] - target) < 1 else 0)

    print(sample_size, train, test, nbr_neuron, "| taux =>", res / test, file=sys.stderr, flush=True)

    return model


def prediction(image, model):
    """ image doit etre de taille 96x96 en noir et blanc"""
    # cv2.resize(image, (96, 96))

    for layer in ConvolutionPart:
        image = layer.prediction(image)
    p = model.prediction(image)

    print(("Healthy" if p > 0 else "Sick"), "with a probability of :", (abs(p) / 2) + 0.5, "%")


def prediction_console(model):

    path = input()
    image = cv2.imread(path,0)
    image = cv2.resize(image,(96,96))
    prediction(np.asarray(image),model)


def plot_training(train, test, nbr_neuron=6, iteration=100):
    data = Data.dataset_tolist()
    sample_size = len(data)
    random.shuffle(data)

    for i in tqdm(range(sample_size)):
        for layer in ConvolutionPart:
            data[i][0] = layer.prediction(data[i][0])

    model = Network()
    model.add(Dense(nbr_neuron, Neuron(21 * 21 * 2, learning_rate=0.01, activation=tanh, dactivation=dtanh)))
    model.add(Dense(1, Neuron(nbr_neuron, learning_rate=0.01, activation=tanh, dactivation=dtanh)))

    # TRAIN

    data_train = data[:train]
    data_test = data[train:train + test]

    y = []
    z = []

    for _ in range(100):
        model.train(data_train, 1)

        # TEST DATASET

        res = 0
        loss = 0
        for data_temp, target in data_test:
            res += (1 if (cost := abs(model.prediction(data_temp)[0] - target)) < 1 else 0)
            loss += cost

        y.append(res/test)
        z.append(loss/(test*2))

    print(y)
    plt.plot(list(range(len(y))), y)
    plt.show()
    print(z)
    plt.plot(list(range(len(z))), z)
    plt.show()

    # print(sample_size, train, test, nbr_neuron, "| taux =>", res / test, file=sys.stderr, flush=True)


plot_training(2000, 2000)

model = training(2000,2000)
while(True):
    prediction_console(model)
