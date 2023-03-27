import time

import matplotlib.pyplot as plt

from FeatureExtraction import FeatureExtraction
from convolution import convolution, reLU, pooling, Kernel_3x3, between_0_1, flatten
from NeuralNewtork import Neuron
import cv2
import numpy as np

# ====================sobel_horizontal===================#
sobel_horizontal =FeatureExtraction() #48x48

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

def get_train_dataset(feature,number):
    train_dataset = []
    targets = []

    # 1 = pizza // 0 = avion
    for i in range(number):
        image = cv2.imread(f"C:\\Users\\Micka\\Documents\\CNN\\Dataset\\96\\avion_{i+1}.jpg", 0)
        image = np.asarray(image)
        image = feature.execute(image)
        train_dataset.append(image)
        targets.append(0)

        image = cv2.imread(f"C:\\Users\\Micka\\Documents\\CNN\\Dataset\\96\\pizza_{i+1}.jpg", 0)
        image = np.asarray(image)
        image = feature.execute(image)
        train_dataset.append(image)
        targets.append(1)

    return train_dataset, targets

def get_train_dataset_second_layer(number,*liste_neuron_feature):
    train_dataset = []
    targets = []

    # 1 = pizza // 0 = avion
    for i in range(number):
        image = cv2.imread(f"C:\\Users\\Micka\\Documents\\CNN\\Dataset\\96\\avion_{i+1}.jpg", 0)
        image = np.asarray(image)
        output_part = []
        for neur_feat in liste_neuron_feature:
            output_part.append(neur_feat[0].prediction(neur_feat[1].execute(image)))
        train_dataset.append(output_part)
        targets.append(0)

        image = cv2.imread(f"C:\\Users\\Micka\\Documents\\CNN\\Dataset\\96\\pizza_{i+1}.jpg", 0)
        image = np.asarray(image)
        output_part = []
        for neur_feat in liste_neuron_feature:
            output_part.append(neur_feat[0].prediction(neur_feat[1].execute(image)))
        train_dataset.append(output_part)
        targets.append(1)

    return train_dataset, targets

def train(feature,neuron_input,number,iteration):
    train_dataset, targets = get_train_dataset(feature,number)
    neuron_input.training(train_dataset,targets,iteration)

def train_second_layer(neuron_to_train,iteration,train_dataset,targets):
    neuron_to_train.training(train_dataset, targets, iteration)

def test(feature,neuron,number_start):
    correcte = 0
    correcte_avion=0
    correcte_pizza=0
    for i in range(50):
        image = cv2.imread(f"C:\\Users\\Micka\\Documents\\CNN\\Dataset\\96\\avion_{number_start+i + 1}.jpg", 0)
        image = np.asarray(image)
        image = feature.execute(image)
        if neuron.prediction(image)<0.5:
            correcte+=1
            correcte_avion+=1

    for i in range(50):
        image = cv2.imread(f"C:\\Users\\Micka\\Documents\\CNN\\Dataset\\96\\pizza_{number_start+i + 1}.jpg", 0)
        image = np.asarray(image)
        image = feature.execute(image)
        if neuron.prediction(image)>0.5:
            correcte+=1
            correcte_pizza+=1

    print("taux =",correcte,"%")
    print("avion =",(correcte_avion/50)*100,"%")
    print("pizza =",(correcte_pizza/50)*100,"%")

def test_2_layer(number_start,neuron_output,*liste_neuron_feature):
    correcte = 0
    correcte_avion=0
    correcte_pizza=0
    for i in range(50):
        image = cv2.imread(f"C:\\Users\\Micka\\Documents\\CNN\\Dataset\\96\\avion_{number_start+i + 1}.jpg", 0)
        image = np.asarray(image)
        output_part = []
        for neur_feat in liste_neuron_feature:
            output_part.append(neur_feat[0].prediction(neur_feat[1].execute(image)))
        if neuron_output.prediction(output_part)<0.5:
            correcte+=1
            correcte_avion+=1

    for i in range(50):
        image = cv2.imread(f"C:\\Users\\Micka\\Documents\\CNN\\Dataset\\96\\pizza_{number_start+i + 1}.jpg", 0)
        image = np.asarray(image)
        output_part = []
        for neur_feat in liste_neuron_feature:
            output_part.append(neur_feat[0].prediction(neur_feat[1].execute(image)))
        if neuron_output.prediction(output_part) >0.5:
            correcte+=1
            correcte_pizza+=1

    print("taux =",correcte,"%")
    print("avion =",(correcte_avion/50)*100,"%")
    print("pizza =",(correcte_pizza/50)*100,"%")




def init():
    debut = time.time()
    #iteration_evaluator()
    second_layer_evaluator()
    print("end in", time.time() - debut)


def second_layer_evaluator():
    neuron_hor = Neuron(np.zeros(21*21))
    neuron_ver = Neuron(np.zeros(9*9))
    neuron_ctr = Neuron(np.zeros(21*21))
    neuron_blr = Neuron(np.zeros(21*21))
    neuron_output = Neuron(np.zeros(4),0,1)



    print("horiziontal")
    train(sobel_horizontal, neuron_hor, 50, 10)
    test(sobel_horizontal, neuron_hor, 100)
    print("vertical")
    train(sobel_vertical, neuron_ver, 100, 10)
    test(sobel_vertical, neuron_ver, 100)
    print("contour")
    train(contour, neuron_ctr, 50, 50)
    test(contour, neuron_ctr, 100)
    print("blur")
    train(blur, neuron_blr, 50, 1000)
    test(blur, neuron_blr, 100)

    train_dataset, targets = get_train_dataset_second_layer(100, [neuron_hor, sobel_horizontal], [neuron_ver, sobel_vertical],[neuron_ctr, contour],[neuron_blr,blur])

    print("2nd layer")
    for i in range(100):
        print(f"\n==========\n{i+1} it")
        train_second_layer(neuron_output,1,train_dataset,targets)
        test_2_layer(100, neuron_output, [neuron_hor, sobel_horizontal], [neuron_ver, sobel_vertical],[neuron_ctr, contour],[neuron_blr,blur])


def iteration_evaluator():
    neuron_hor = Neuron(np.zeros(21*21))
    neuron_ver = Neuron(np.zeros(21*21))
    neuron_ctr = Neuron(np.zeros(21*21))
    neuron_blr = Neuron(np.zeros(21*21))

    ntr, nte = 25, 100

    print("\n==========\n10 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor,ntr,10)
    test(sobel_horizontal,neuron_hor,nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,10)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 10)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 10)
    test(blur, neuron_blr, nte)


    print("\n==========\nnt it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 40)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,40)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 40)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 40)
    test(blur, neuron_blr, nte)


    print("\n==========\n100 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 50)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,50)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 50)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 50)
    test(blur, neuron_blr, nte)

    print("\n==========\n200 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 100)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,100)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 100)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 100)
    test(blur, neuron_blr, nte)

    print("\n==========\n500 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 300)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,300)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 300)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 300)
    test(blur, neuron_blr, nte)

    print("\n==========\n1000 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 500)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,500)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 500)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 500)
    test(blur, neuron_blr, nte)

    print("\n==========\n1500 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 500)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,500)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 500)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 500)
    test(blur, neuron_blr, nte)

    print("\n==========\n2000 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 500)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,500)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 500)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 500)
    test(blur, neuron_blr, nte)

    print("\n==========\n3000 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 1000)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,1000)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 1000)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 1000)
    test(blur, neuron_blr, nte)

    print("\n==========\n5000 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 2000)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,2000)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 2000)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 2000)
    test(blur, neuron_blr, nte)

    print("\n==========\n10000 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 5000)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,5000)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 5000)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 5000)
    test(blur, neuron_blr, nte)

    print("\n==========\n50000 it")
    print("horiziontal")
    train(sobel_horizontal,neuron_hor, ntr, 40000)
    test(sobel_horizontal,neuron_hor, nte)
    print("vertical")
    train(sobel_vertical,neuron_ver,ntr,40000)
    test(sobel_vertical,neuron_ver,nte)
    print("contour")
    train(contour, neuron_ctr, ntr, 40000)
    test(contour, neuron_ctr, nte)
    print("blur")
    train(blur, neuron_blr, ntr, 40000)
    test(blur, neuron_blr, nte)



if __name__ == '__main__':
    init()
