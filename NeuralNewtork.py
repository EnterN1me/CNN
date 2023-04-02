import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    # equivalent a la dérivé et plus simple
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def dtanh(x):
    return 1 - (tanh(x) ** 2)


class Neuron:

    def __init__(self, weights: np.ndarray, bias=0., learning_rate=0.1, activation=sigmoid, dactivation=dsigmoid, possible_value=[0,1]):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.activation = activation
        self.dactivation = dactivation
        self.possible_value= possible_value
        self.ecart = abs(possible_value[0]-possible_value[1])

    def prediction(self, vector: np.ndarray) -> float:
        prediction = np.dot(vector, self.weights) + self.bias  # np.dot est le produit scalaire
        prediction = self.activation(prediction)
        return prediction

    def gradient(self, vector: np.ndarray, target):
        layer = np.dot(vector, self.weights) + self.bias
        prediction = self.activation(layer)
        erreur = (prediction - target) ** 2

        # derivé : tout les calcul ont été verifié par moi meme (derivé du premier par rapport au deuxieme)
        derreur_dprediction = 2 * (prediction - target)
        dlayer_dweights = vector
        dlayer_dbias = 1

        dprediction_dlayer = self.dactivation(layer)  # vrai pour toute fonction

        derreur_dweights = derreur_dprediction * np.multiply(dprediction_dlayer, dlayer_dweights)
        derreur_dbias = derreur_dprediction * dprediction_dlayer * dlayer_dbias

        return derreur_dweights, derreur_dbias

    def update_parametre(self, derreur_dweights, derreur_dbias):
        self.weights = self.weights - np.multiply(self.learning_rate, derreur_dweights)
        self.bias = self.bias - (self.learning_rate * derreur_dbias)

    def training(self, vectors: list[np.ndarray], targets: list[float], iterations=1):
        for iteration in range(iterations):
            for i in range(len(vectors)):
                input_vector = vectors[i]
                input_target = targets[i]

                derreur_dweights, derreur_dbias = self.gradient(input_vector, input_target)

                self.update_parametre(derreur_dweights, derreur_dbias)

    # PLOT METHOD

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
                error += abs(prediction - test_actuel[1])/self.ecart
                bad += (1 if abs(prediction - test_actuel[1]) >= self.ecart/2 else 0)
            list_errors.append(error*100 / len(test))
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

    # SAVE METHOD

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
