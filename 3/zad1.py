import random
import time

import idx2numpy
import numpy as np
from sklearn import preprocessing

import NeuralNetwork_Class as nn
import zad3_class as nn3


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def test():
    network = nn.NeuralNetwork(1)
    network.load_weights_form_txt("weights_test")
    x = np.matrix([[0.5, 0.1, 0.2, 0.8, ], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
    y = np.matrix([[0.1, 0.5, 0.1, 0.7], [1.0, 0.2, 0.3, 0.6], [0.1, -0.5, 0.2, 0.2]])
    alpha = 0.01
    print(network.weights)
    network.neural_network_learn(x[:, 0], y[:, 0], alpha)
    print(network.weights)


def zad1():
    network = nn.NeuralNetwork(1)
    network.load_weights_form_txt("weights_zad1")
    x = np.matrix([[0.5, 0.1, 0.2, 0.8], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
    res = network.neural_network_first(x)
    num = 1
    for i in res:
        print(f"wyjście sieci po {num} serii:\n {i}")
        num += 1


def zad2():
    network = nn.NeuralNetwork(1)
    x = np.matrix([[0.5, 0.1, 0.2, 0.8], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
    y = np.matrix([[0.1, 0.5, 0.1, 0.7], [1.0, 0.2, 0.3, 0.6], [0.1, -0.5, 0.2, 0.2]])

    network.load_weights_form_txt("weights_zad2")
    res = network.neural_network_learn(x, y, 0.01)
    print(color.BOLD + color.UNDERLINE + "\nPo 1 epoce:" + color.END)
    num = 1
    for i in res:
        print(f"wyjście sieci po {num} serii:\n {i}")
        num += 1
    network.load_weights_form_txt("weights_zad2")
    for i in range(50):
        res = network.neural_network_learn(x, y, 0.01)
    num = 1
    print(color.BOLD + color.UNDERLINE + "\nPo 50 epoce:" + color.END)
    for i in res:
        print(f"wyjście sieci po {num} serii:\n {i}")
        num += 1


def random_generator():
    l = [random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(0, 10)]
    while l[0] > l[1]:
        l = [random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(0, 10)]
    return l


def create_matrix_from_label(l):
    m = np.asmatrix(np.zeros((10, 1)))
    m[l][0] = 1
    return m


def get_labels(lb, size):
    l = []
    for i in range(size):
        l.append(create_matrix_from_label(lb[i]))
    return l


def zad3():
    label_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    picture_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')

    label_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    picture_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')

    label_test = get_labels(label_test, 10000)
    label_train = get_labels(label_train, 60000)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scale_train = scaler.fit_transform(picture_train.reshape(-1, picture_train.shape[-1])).reshape(picture_train.shape)
    scale_test = scaler.fit_transform(picture_test.reshape(-1, picture_test.shape[-1])).reshape(picture_test.shape)

    picture_train = scale_train.reshape((60000, 784, 1))
    picture_test = scale_test.reshape((10000, 784, 1))

    network = nn3.NeuralNetwork(784)
    network.add_layer(40, -0.01, 0.01)
    network.add_layer(10, -0.01, 0.01)

    print("Start")
    for i in range(2):
        print(color.BOLD + color.UNDERLINE + f"\nIteracja {i + 1}:" + color.END)
        how_many = 0

        network.fit(picture_train, label_train)
        for j in range(10_000):
            if np.argmax(network.predict(picture_test[j])) == np.argmax(label_test[j]):
                how_many += 1
        print(f"Skuteczność sieci - test: {how_many / 10_000:.4%}")

        how_many = 0
        for j in range(60_000):
            if np.argmax(network.predict(picture_train[j])) == np.argmax(label_train[j]):
                how_many += 1
        print(f"Skuteczność sieci - train: {how_many / 60_000:.4%}")

    network.save_weights("zad_4_weights")


def zad3_test():
    label_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    picture_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')

    label_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    picture_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')

    label_test = get_labels(label_test, 10000)
    label_train = get_labels(label_train, 60000)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scale_train = scaler.fit_transform(picture_train.reshape(-1, picture_train.shape[-1])).reshape(picture_train.shape)
    scale_test = scaler.fit_transform(picture_test.reshape(-1, picture_test.shape[-1])).reshape(picture_test.shape)

    picture_train = scale_train.reshape((60000, 784, 1))
    picture_test = scale_test.reshape((10000, 784, 1))

    network = nn3.NeuralNetwork(784)
    network.load_weights("zad_4_weights", 2)

    how_many = 0
    for j in range(10_000):
        if np.argmax(network.predict(picture_test[j])) == np.argmax(label_test[j]):
            how_many += 1
    print(f"Skuteczność sieci - test: {how_many / 10_000:.4%}")

    how_many = 0
    for j in range(60_000):
        if np.argmax(network.predict(picture_train[j])) == np.argmax(label_train[j]):
            how_many += 1
    print(f"Skuteczność sieci - train: {how_many / 60_000:.4%}")


def zad4():
    mini, max, alpha = random_generator()
    network = nn.NeuralNetwork(3)
    network.add_layer(5, -0.01, 0.01)
    network.add_layer(4, -0.01, 0.01)

    # network.load_weights_form_txt("weights_zad4")

    rgb_train = []
    colors_train = []
    network.load_colors("training_colors", rgb_train, colors_train)
    rgb_test = []
    colors_test = []
    network.load_colors("test_colors", rgb_test, colors_test)

    how_many = 0
    times = 1
    while True:
        how_many = 0
        for i in range(130):
            predicted = network.predict(rgb_test[i])
            if np.argmax(predicted) == np.argmax(colors_test[i]):
                how_many += 1
        print(f"Skuteczność sieci: {how_many / 130:.4%} za {times} razem")
        if (how_many == 130):
            network.save_weights("zad_3_weights")
            break
        for j in range(109):
            network.neural_network_learn(rgb_train[j], colors_train[j], 0.05)
        times += 1


def main():
    print(f"\nZadanie 1 start\n")
    zad1()
    print(f"\nZadanie 1 stop\n")

    print(f"\nZadanie 2 start\n")
    zad2()
    print(f"\nZadanie 2 stop\n")

    print(f"\nZadanie 4 start\n")
    zad4()
    print(f"\nZadanie 4 stop\n")

    print(f"\nZadanie 3 start\n")
    start_time = time.time()
    zad3()
    end_time = time.time()
    print('\nDuration sec: {}'.format(end_time - start_time))
    print('Duration min: {}'.format((end_time - start_time) / 60))
    print(f"\nZadanie 3 stop\n")

    print(f"\nZadanie 3 start - ładnowanie wag z pliku\n")
    zad3_test()
    print(f"\nZadanie 3 stop - ładnowanie wag z pliku\n")


if __name__ == "__main__":
    main()
