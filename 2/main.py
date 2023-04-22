import numpy as np
import class_file as nn


def input(input, weights, bias=0):
    neuron_sum = 0

    for i in range(len(input)):
        neuron_sum += (input[i] * weights[i][0])
    return neuron_sum + bias


def neural_network(weights, input):
    return np.dot(weights, input)


def deep_neural_network(weights, input):
    value = neural_network(weights[0], input)
    for i in range(1, len(weights)):
        value = neural_network(weights[i], value)
    return value


def simple_learn(w, x, y, alpha, num):
    for i in range(num - 1):
        output = w * x
        delta = np.dot(np.dot(2, np.subtract(output, y)), x)
        w = np.subtract(w, np.dot(delta, alpha))

    err = np.square((np.subtract(np.dot(w, x), y)))
    output = np.dot(w, x)

    print("Output: ", output)
    print("Error: ", err)


def zad1():
    print("Zadanie pierwsze:")
    w = 0.5
    y = 0.8
    x = 2
    alpha = 0.1
    simple_learn(w, x, y, alpha, 5)
    simple_learn(w, x, y, alpha, 20)


def zad2():
    print("\nZadanie drugie:")
    network = nn.NeuralNetwork()
    network.load_weights("weights")

    y = np.matrix([
        [0.1, 0.5, 0.1, 0.7],
        [1.0, 0.2, 0.3, 0.6],
        [0.1, -0.5, 0.2, 0.2],
        [0.0, 0.3, 0.9, -0.1],
        [-0.1, 0.7, 0.1, 0.8]])
    x = np.matrix([[0.5, 0.1, 0.2, 0.8],
                   [0.75, 0.3, 0.1, 0.9],
                   [0.1, 0.7, 0.6, 0.2]])

    alpha = 0.01

    for i in range(1000):
        network.neural_network_smart(x, y, alpha)

    print(np.sum(network.err))

def zad3():
    print("\nZadanie trzecie:")
    network = nn.NeuralNetwork()
    network.load_weights("weights2")

    rgb_train = []
    colors_train = []
    network.load_colors("training_colors", rgb_train, colors_train)
    rgb_test = []
    colors_test = []
    network.load_colors("test_colors", rgb_test, colors_test)

    how_many = 0
    while how_many != 130:
        how_many = 0
        for i in range(130):
            predicted = network.predict(rgb_test[i])
            if np.argmax(predicted) == np.argmax(colors_test[i]):
                how_many += 1
        print(f"Skuteczność sieci: {how_many / 130:.4%}")
        for j in range(109):
            network.neural_network_smart(rgb_train[j], colors_train[j], 0.01)









    # network.neural_network_smart(rgb[0],colors[0],0.01)


def main():
    zad1()
    zad2()
    zad3()


if __name__ == '__main__':
    main()

#%%
