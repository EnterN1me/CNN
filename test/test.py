import numpy as np
import matplotlib.pyplot as plt

import NeuralNetwork


def test_linear():

    # basic and calculus # =========================== #
    init_weight_number = 1
    init_bias = 1
    neuron = NeuralNetwork.Lineaire(init_weight_number, init_bias, 0.01)
    assert neuron.prediction(2) == 1, "prediction error : " + str(neuron.prediction(2)) + str(neuron.weights) + str(
        neuron.bias)
    neuron.learning(2, 4)
    assert neuron.weights == np.zeros(init_weight_number) - np.multiply(neuron.learning_rate, (2 * (0 * 2 + 1 - 4)))  # 1.2
    assert neuron.bias == init_bias - neuron.learning_rate * (0 * 2 + 1 - 4)
    assert neuron.prediction(2) > 1
    del neuron

    # random weights # =========================== #
    neuron = NeuralNetwork.Lineaire()
    neuron.set_random_weights()
    assert neuron.weights!=[0], "random initialisation error "+str(neuron.weights)
    del neuron

    # 2 layer network
    hidden_neuron = [NeuralNetwork.Lineaire(1,0,0.001) for _ in range(4)]
    for neuron in hidden_neuron:
        neuron.set_random_weights(20)
    output_neuron = NeuralNetwork.Lineaire(4,0,0.00001)
    output_neuron.set_random_weights(15)

    # layer 1
    input_value = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    target_value = [1.9,1.4,1.3,1.2,0.9,0.6,0.5,0.4,0.2,0.3,0.1,0.5,0.6,0.7,0.8,0.9,1.0,1.4,1.6,1.8,1.9]

    for _ in range(100):

        for i in range(len(input_value)):
            prediction = [neuron.prediction(input_value[i]) for neuron in hidden_neuron]
            for neuron in hidden_neuron:
                neuron.learning(input_value[i],target_value[i])
            output_neuron.learning(prediction,target_value[i])

    plot = []
    for z in list(range(21)):
        prediction = [neuron.prediction(z) for neuron in hidden_neuron]
        plot.append(output_neuron.prediction(prediction))

    plt.plot(list(range(21)), plot, 'r')
    for neuron in hidden_neuron:
        plt.plot(list(range(21)), [neuron.prediction(i) for i in list(range(21))])
        print(neuron.weights)
    plt.scatter(input_value, target_value)
    plt.show()










if __name__ == "__main__":
    test_linear()
