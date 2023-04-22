import numpy as np
from keras.datasets import mnist
from sklearn import preprocessing

import NeuralNetwork as Neural


def zad1():
    network = Neural.NeuralNetwork(784)
    input_image = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    filter = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    res = network.conv_zad1(input_image,filter)

    print(res)


def zad2():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    y_train = np.zeros((train_y.shape[0], train_y.max() + 1), dtype=np.float32)
    y_train[np.arange(train_y.shape[0]), train_y] = 1
    y_test = np.zeros((test_y.shape[0], test_y.max() + 1), dtype=np.float32)
    y_test[np.arange(test_y.shape[0]), test_y] = 1

    # test_X = test_X.reshape(10000, 784)
    # train_X = train_X.reshape(60000, 784)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scale_train = scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    scale_test = scaler.fit_transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    test_X = scale_test
    train_X = scale_train

    network = Neural.NeuralNetwork(10816)
    network.add_layer(10, -0.1, 0.1)
    filter = network.add_filters(-0.01, 0.01, 3, 3, 16)


    # network.predict_zad2(train_X[0],filter, y_train[0])

    # network.fit_zad2(train_X[0:10], y_train[0:10], filter, 0.01, 10)
    # network.save_weights("zad_2", filter)


    print("Zad 2 - 60_000/10_000")
    for i in range(50):
        network.fit_zad2(train_X[0:60_000], y_train[0:60_000], filter, 0.01, 10)
        how_many = 0
        for i in range(10_000):
            how_many = how_many + network.predict_zad2(test_X[i], filter, y_test[i])

        print(f"\nSkutecznosc sieci - test: {how_many / 10_000:.4%}\n")

    network.save_weights("zad2_3", filter)





def zad3():
    # network = Neural.NeuralNetwork(10816)
    # input = np.array(
    #     [[3, 6, 7, 5, 3, 5], [6, 2, 9, 1, 2, 7], [0, 9, 3, 6, 0, 6], [2, 6, 1, 8, 7, 9], [2, 0, 2, 3, 7, 5],
    #      [9, 2, 2, 8, 9, 7]])
    # network.pooling(input, 2, 2)

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    y_train = np.zeros((train_y.shape[0], train_y.max() + 1), dtype=np.float32)
    y_train[np.arange(train_y.shape[0]), train_y] = 1
    y_test = np.zeros((test_y.shape[0], test_y.max() + 1), dtype=np.float32)
    y_test[np.arange(test_y.shape[0]), test_y] = 1

    # test_X = test_X.reshape(10000, 784)
    # train_X = train_X.reshape(60000, 784)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scale_train = scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    scale_test = scaler.fit_transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    test_X = scale_test
    train_X = scale_train

    network = Neural.NeuralNetwork(2704)
    network.add_layer(10, -0.1, 0.1)
    filter = network.add_filters(-0.01, 0.01, 3, 3, 16)


    print("Zad 3 - 1_000/10_000")
    for i in range(5):
        network.fit_zad3(train_X[0:1_00], y_train[0:1_00], filter, 0.001, 10, 1_00)
        how_many = 0
        for i in range(10_000):
            how_many = how_many + network.predict_zad3(test_X[i], filter, y_test[i])

        print(f"\nSkutecznosc sieci - test: {how_many / 10_000:.4%}\n")

    network.save_weights("zad3_3", filter)



def main():
    # zad1()
    # zad2()
    zad3()

if __name__ == '__main__':
    main()
