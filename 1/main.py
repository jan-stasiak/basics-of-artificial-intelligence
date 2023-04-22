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


def main():
    # wagi z przykładu 1.2.2 wczytane z pliku w formacie (struktura pliku tekstowego - macierz,#,macierz,#)
    network = nn.NeuralNetwork()
    network.load_weights("weights")

    # potwierdzenie poprawności wczytywania wag oraz działania funkcji deep_neural_network poprzez uzyskanie wyników zgodnych z wynikami ze skryptu
    x = np.array([[0.5], [0.75], [0.1]])
    print(deep_neural_network(network.weights, x))


if __name__ == '__main__':
    main()
