import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    # equivalent a la dérivé et plus simple
    return sigmoid(x) * (1 - sigmoid(x))


class Neuron:

    def __init__(self, weights, bias=0, learning_rate=0.1):
        self._weights = weights
        self._bias = bias
        self._learning_rate = learning_rate

    def prediction(self, vector):
        prediction = np.dot(vector, self._weights) + self._bias  # np.dot est le produit scalaire
        prediction = sigmoid(prediction)
        return prediction

    def gradient(self, vector, target):
        layer = np.dot(vector, self._weights) + self._bias
        prediction = sigmoid(layer)
        erreur = (prediction - target) ** 2

        # derivé : tout les calcul ont été verifié par moi meme (derivé du premier par rapport au deuxieme)
        derreur_dprediction = 2 * (prediction - target)
        dlayer_dweights = vector
        dlayer_dbias = 1

        dprediction_dlayer = dsigmoid(layer)
        derreur_dweights = np.multiply(derreur_dprediction, np.multiply(dprediction_dlayer, dlayer_dweights))
        derreur_dbias = derreur_dprediction * dprediction_dlayer * dlayer_dbias

        return derreur_dweights, derreur_dbias

    def update_parametre(self, derreur_dweights, derreur_dbias):
        self._weights = self._weights - np.multiply(self._learning_rate, derreur_dweights)
        self._bias = self._bias - (self._learning_rate * derreur_dbias)

    def training(self, vectors, targets, iterations=1):
        for iteration in range(iterations):
            for i in range(len(vectors)):
                input_vector = vectors[i]
                input_target = targets[i]

                derreur_dweights, derreur_dbias = self.gradient(input_vector, input_target)

                self.update_parametre(derreur_dweights, derreur_dbias)

    def plot_weight(self):
        cote = round(np.sqrt(self._weights.shape[0]))
        montrer = [[self._weights[i+j] for j in range(cote)] for i in range(0, round(self._weights.shape[0]), cote)]
        montrer = np.asarray(montrer)

        plt.imshow(montrer)
        plt.axis('off')
        plt.savefig('image.png', bbox_inches='tight', pad_inches=0)
        plt.show()