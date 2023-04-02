import time

import matplotlib.pyplot as plt

from FeatureExtraction import FeatureExtraction
from convolution import convolution, reLU, pooling, Kernel_3x3, between_0_1, flatten
from NeuralNewtork import Neuron, tanh, dtanh
import cv2
import numpy as np
from Data import Dataset

# ====================sobel_horizontal===================#
sobel_horizontal = FeatureExtraction() #48x48

sobel_horizontal.add_layer(convolution, Kernel_3x3["sobel_horizontal"]) #46x46
sobel_horizontal.add_layer(reLU)

sobel_horizontal.add_layer(pooling, 2) #23x23

sobel_horizontal.add_layer(convolution, Kernel_3x3["sobel_horizontal"]) #21x21
sobel_horizontal.add_layer(reLU)

sobel_horizontal.add_layer(pooling, 2) #11x11

sobel_horizontal.add_layer(convolution, Kernel_3x3["sobel_horizontal"]) #9x9
sobel_horizontal.add_layer(reLU)

sobel_horizontal.add_layer(between_0_1)
sobel_horizontal.add_layer(flatten)


 #====================sobel_vertical===================#
sobel_vertical =FeatureExtraction() #48x48

sobel_vertical.add_layer(convolution, Kernel_3x3["sobel_vertical"]) #46x46
sobel_vertical.add_layer(reLU)

sobel_vertical.add_layer(pooling, 2) #23x23

sobel_vertical.add_layer(convolution, Kernel_3x3["sobel_vertical"]) #21x21
sobel_vertical.add_layer(reLU)

sobel_vertical.add_layer(pooling, 2) #11x11

sobel_vertical.add_layer(convolution, Kernel_3x3["sobel_vertical"]) #9x9
sobel_vertical.add_layer(reLU)

sobel_vertical.add_layer(between_0_1)
sobel_vertical.add_layer(flatten)

 #====================contour===================#
contour =FeatureExtraction() #48x48

contour.add_layer(convolution, Kernel_3x3["contour"]) #46x46
contour.add_layer(reLU)

contour.add_layer(pooling, 2) #23x23

contour.add_layer(convolution, Kernel_3x3["contour"]) #21x21
contour.add_layer(reLU)

contour.add_layer(pooling, 2) #11x11

contour.add_layer(convolution, Kernel_3x3["contour"]) #9x9
contour.add_layer(reLU)

contour.add_layer(between_0_1)
contour.add_layer(flatten)

 #====================blur===================#
blur =FeatureExtraction() #48x48

blur.add_layer(convolution, Kernel_3x3["blur"]) #46x46
blur.add_layer(reLU)

blur.add_layer(pooling, 2) #23x23

blur.add_layer(convolution, Kernel_3x3["blur"]) #21x21
blur.add_layer(reLU)

blur.add_layer(pooling, 2) #11x11

blur.add_layer(convolution, Kernel_3x3["blur"]) #9x9
blur.add_layer(reLU)

blur.add_layer(between_0_1)
blur.add_layer(flatten)


# pizza = 1 // avion = -1
def plot_training():
    neuron_hor = Neuron(np.zeros(21 * 21), learning_rate=0.05, activation=tanh, dactivation=dtanh, possible_value=[-1, 1])

    Pizza = Dataset("Dataset\\trait")
    pizza_data = Pizza.get_new_dataset(24,"trait","jpg",1)
    pizza_data = Pizza.add_feature_to_dataset(pizza_data,sobel_horizontal)
    Avion = Dataset("Dataset\\trait")
    avion_data = Avion.get_new_dataset(24,"sans","jpg",-1)
    avion_data = Avion.add_feature_to_dataset(avion_data,sobel_horizontal)

    dt_plt = []
    dt_avt = []
    data = []
    for i in range(10):
        data.append(pizza_data[i])
        data.append(avion_data[i])
        dt_plt.append(pizza_data[i][0])
        dt_avt.append(avion_data[i][0])

    data_test = []
    for i in range(10, 20):
        data_test.append(pizza_data[i])
        data_test.append(avion_data[i])

    neuron_hor.plot_training_error(data, data_test, 1000, 5)


plot_training()


