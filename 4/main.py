import time

import idx2numpy
import numpy as np
import NeuralNetwork as Neural
import random
from sklearn import preprocessing
from Colors import color
from keras.datasets import mnist
import silence_tensorflow.auto

def generate_ones(x):
    for i in range(int(x.shape[0] / 2)):
        x[i] = 1
    return x


def shuffle_vector(x):
    return random.shuffle(x)

def create_matrix_from_label(l):
    m = np.asmatrix(np.zeros((10, 1)))
    m[l][0] = 1
    return m


def get_labels(lb, size):
    l = []
    for i in range(size):
        l.append(create_matrix_from_label(lb[i]))
    return l


def zad1():
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

    network = Neural.NeuralNetwork(784)
    network.add_layer(40, -0.1, 0.1)
    network.add_layer(10, -0.1, 0.1)

    for i in range(351):
        how_many = 0
        if i % 10 == 0:
            print(color.BOLD + color.UNDERLINE + f"\nIteracja {i}:" + color.END)
            for j in range(10_000):
                if np.argmax(network.predict(picture_test[j])) == np.argmax(label_test[j]):
                    how_many += 1
            print(f"Skutecznosc sieci - test: {how_many / 10_000:.4%}")

            how_many = 0
            for j in range(1_000):
                if np.argmax(network.predict(picture_train[j])) == np.argmax(label_train[j]):
                    how_many += 1
            print(f"Skutecznosc sieci - train: {how_many / 1_000:.4%}")
        network.fit(picture_train, label_train)


    network.save_weights("zad_1_weights")


def zad2():
    # label_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    # picture_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
    #
    # label_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    # picture_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
    #
    # label_test = get_labels(label_test, 10000)
    # label_train = get_labels(label_train, 60000)
    #
    #
    # picture_train = scale_train.reshape((60000, 784, 1))
    # picture_test = scale_test.reshape((10000, 784, 1))

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    y_train = np.zeros((train_y.shape[0], train_y.max() + 1), dtype=np.float32)
    y_train[np.arange(train_y.shape[0]), train_y] = 1
    y_test = np.zeros((test_y.shape[0], test_y.max() + 1), dtype=np.float32)
    y_test[np.arange(test_y.shape[0]), test_y] = 1

    test_X = test_X.reshape(10000, 784)
    train_X = train_X.reshape(60000, 784)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scale_train = scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    scale_test = scaler.fit_transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    test_X = scale_test
    train_X = scale_train

    network = Neural.NeuralNetwork(784)
    network.add_layer(100, -0.1, 0.1)
    network.add_layer(10, -0.1, 0.1)

    print("Test zadania 2")
    for i in range(350):
        how_many = 0
        for j in range(10_000):
            if np.argmax(network.predict(test_X[j])) == test_y[j]:
                how_many += 1
        print(f"Skutecznosc sieci - test {i + 1}: {how_many / 10_000:.4%}")
        network.fit_batch(train_X, y_train, 100, 1)



def zad3():
    # label_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    # picture_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
    #
    # label_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    # picture_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
    #
    # label_test = get_labels(label_test, 10000)
    # label_train = get_labels(label_train, 60000)
    #
    # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # scale_train = scaler.fit_transform(picture_train.reshape(-1, picture_train.shape[-1])).reshape(picture_train.shape)
    # scale_test = scaler.fit_transform(picture_test.reshape(-1, picture_test.shape[-1])).reshape(picture_test.shape)
    #
    # picture_train = scale_train.reshape((60000, 784, 1))
    # picture_test = scale_test.reshape((10000, 784, 1))

    # (train_X, train_y), (test_X, test_y) = mnist.load_data()

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    y_train = np.zeros((train_y.shape[0], train_y.max() + 1), dtype=np.float32)
    y_train[np.arange(train_y.shape[0]), train_y] = 1
    y_test = np.zeros((test_y.shape[0], test_y.max() + 1), dtype=np.float32)
    y_test[np.arange(test_y.shape[0]), test_y] = 1

    test_X = test_X.reshape(10000, 784)
    train_X = train_X.reshape(60000, 784)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scale_train = scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    scale_test = scaler.fit_transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    test_X = scale_test
    train_X = scale_train

    network = Neural.NeuralNetwork(784)
    network.add_layer(100, -0.01, 0.01)
    network.add_layer(10, -0.01, 0.01)

    print("Test zadania 3. Dla 10000")
    for i in range(350):
        how_many = 0
        for j in range(10_000):
            if np.argmax(network.predict_fun(test_X[j])) == test_y[j]:
                how_many += 1
        print(f"Skutecznosc sieci - test {i}: {how_many / 10_000:.4%}")
        network.fit_batch_fun(train_X, y_train, 100, 1)







def main():
    # print(f"Alpha: 0.005, 1000 training set, 10000 test set, weights <-0.1, 0.1>, 40 hiddens, dropout")
    # start = time.time()
    # zad1()
    # stop = time.time()
    # print(f"\nOperation took: {stop - start} seconds")

    # start = time.time()
    # zad2()
    # stop = time.time()
    # print(f"\nOperation took: {stop - start} seconds")

    start = time.time()
    zad3()
    stop = time.time()
    print(f"\nOperation took: {stop - start} seconds")


if __name__ == '__main__':
    main()