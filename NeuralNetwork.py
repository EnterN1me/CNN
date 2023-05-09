import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod


class BaseNeuron(ABC):

    @abstractmethod
    def prediction(self, data) -> float:
        pass

    @abstractmethod
    def gradient(self, data, target) -> tuple:
        pass

    @abstractmethod
    def learning(self) -> None:
        pass


class Lineaire(BaseNeuron):

    def __init__(self, weights_number=1, bias=0, learning_rate=0.1):
        self.weights = np.zeros(weights_number)
        self.bias = bias
        self.learning_rate = learning_rate

    def set_random_weights(self,multiplicateur=1):
        self.weights = np.asarray([(np.random.random() - 0.5) * multiplicateur for _ in range(len(self.weights))])

    def prediction(self, data) -> float:
        prediction = np.dot(data, self.weights) + self.bias
        if abs(prediction - int(prediction)) < 0.01:
            return int(prediction)
        return float(prediction)

    def gradient(self, data, target):
        prediction = np.dot(data, self.weights) + self.bias
        # erreur = (prediction - target) ** 2

        derreur_dprediction = (prediction - target)
        dprediction_dweights = data
        dprediction_dbias = 1

        derreur_dweights = np.multiply(derreur_dprediction,dprediction_dweights)
        derreur_dbias = derreur_dprediction * dprediction_dbias

        return derreur_dweights, derreur_dbias

    def learning(self, data, target):
        derreur_dweights, derreur_dbias = self.gradient(data, target)
        self.weights = self.weights - np.multiply(self.learning_rate, derreur_dweights)
        self.bias = self.bias - (self.learning_rate * derreur_dbias)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    # equivalent a la dérivé et plus simple
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def dtanh(x):
    return 1 - (tanh(x) ** 2)


class Neuron(Lineaire):

    def __init__(self, weights: np.ndarray, bias=0., learning_rate=0.1, activation=sigmoid,
                 dactivation=dsigmoid):

        super().__init__(weights,bias,learning_rate)
        self.activation = activation
        self.dactivation = dactivation

    def prediction(self, data: np.ndarray) -> float:
        return self.activation(super().prediction(data))

    def gradient(self, data: np.ndarray, target):

        derreur_dweights, derreur_dbias = super().gradient(data, target)

        dprediction_dlayer = self.dactivation(super().prediction(data))  # vrai pour toute fonction

        derreur_dweights *= dprediction_dlayer
        derreur_dbias *= dprediction_dlayer

        return derreur_dweights, derreur_dbias

    def learning(self, data, target):
        derreur_dweights, derreur_dbias = self.gradient(data, target)
        self.weights = self.weights - np.multiply(self.learning_rate, derreur_dweights)
        self.bias = self.bias - (self.learning_rate * derreur_dbias)

    def training(self, vectors: list[np.ndarray], targets: list[float], iterations=1):
        for iteration in range(iterations):
            for i in range(len(vectors)):
                input_vector = vectors[i]
                input_target = targets[i]

                self.learning(input_vector, input_target)

    # PLOT METHOD #=================================================================================================#

    def plot_training_error(self, train: list[list[np.ndarray, float]], test: list[list[np.ndarray, float]],
                            iteration=100, pas=10):
        train_vectors = [train[i][0] for i in range(len(train))]
        train_targets = [float(train[i][1]) for i in range(len(train))]
        list_errors = []
        bad_prediction = []
        x = []
        for i in range(iteration // pas):
            self.training(train_vectors, train_targets, pas)
            error = 0
            bad = 0
            for test_actuel in test:
                prediction = self.prediction(test_actuel[0])
                error += abs(prediction - test_actuel[1]) / 2
                bad += (1 if abs(prediction - test_actuel[1]) >= 2 / 2 else 0)
            list_errors.append(error * 100 / len(test))
            bad_prediction.append(bad * 100 / len(test))
            x.append(i * pas)
        plt.figure("mean of error in value (%) / bad prediction (%)")
        plt.subplot(1, 2, 1)
        plt.plot(x, list_errors, 'g')
        plt.subplot(1, 2, 2)
        plt.plot(x, bad_prediction, 'r')
        plt.show()

    def plot_weight(self):
        cote = round(np.sqrt(self.weights.shape[0]))
        montrer = [[self.weights[i + j] for j in range(cote)] for i in range(0, round(self.weights.shape[0]), cote)]
        montrer = np.asarray(montrer)

        plt.imshow(montrer)
        plt.axis('off')
        # plt.savefig('image.png', bbox_inches='tight', pad_inches=0)
        plt.show()

    # SAVE METHOD #===============================================================================================#

    def save(self, name):
        """sauvegarde les coefficients et le biais dans un fichier {name}.txt"""

        with open(f"saved_neuron\\{name}.txt", 'w') as file:
            text = str(self.weights) + "\n@\n" + str(self.bias)
            file.write(text)

    def load(self, name):
        """charge les coefficients et le biais du fichier {name}.txt"""

        with open(f"saved_neuron\\{name}.txt", 'r') as file:
            text = ""
            for line in file:
                text += line

            # on retire les saut de ligne et on separe les coeffs du bias
            text = text.replace("\n", "").split('@')  # @ entre weight et bias

            # on prend la liste des coefficients
            temp_weight = [valeur for valeur in text[0].replace("[", "").replace("]", "").split(" ")]

            # y a des double espace qui crée des valeur null a cause du split avec 1 seul epsace
            temp_weight = list(filter(None, temp_weight))

            # on passe les coeffs de str a float64
            temp_weight = list(map(np.float64, temp_weight))

            # on modifie les valeur du neuronne
            self.weights = np.asarray(temp_weight)
            self.bias = np.float64(text[1])  # pas de pb pour ca
