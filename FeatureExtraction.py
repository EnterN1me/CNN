import matplotlib.pyplot as plt
import numpy as np
import convolution


class FeatureExtraction:

    def __init__(self):
        self.layers = []

    def add_layer_with_parametre(self,layer,parametre):

        def conv(image):
            return layer(image,parametre)

        self.add_layer(conv)

    def add_layer(self,layer,parametre=None):
        if type(parametre)!=type(None):
            self.add_layer_with_parametre(layer,parametre)
            return
        self.layers.append(layer)

    def get_last_layer(self):
        return self.layers[-1]

    def execute(self,image_input):
        image = image_input

        for layer in self.layers:
            image = layer(image)

        return image

    # affichage de chaque effectué, on retire les between_0_1 car fait rien de visible et flatten parce que marche pas
    def plot(self,image_input):
        lenght = len(self.layers) - self.layers.count(convolution.flatten) - self.layers.count(convolution.between_0_1)
        sqr = np.sqrt(lenght+1)
        row = int(np.ceil(sqr))
        col = int(np.ceil(sqr))

        plt.figure()
        plt.subplot(row,col,1)
        image = image_input.copy()
        plt.axis('off')
        plt.imshow(image)

        for i,layer in enumerate(self.layers):
            if (layer==convolution.flatten) or (layer==convolution.between_0_1):
                continue

            image = layer(image)
            print(row,col,i+2)
            plt.subplot(row,col,2+i)
            plt.axis('off')
            plt.imshow(image)

        plt.axis('off')
        plt.show()
        return image
