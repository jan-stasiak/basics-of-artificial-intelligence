import os

import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle


class NeuralNetwork:
    def __init__(self, enters=0):
        self.filepath = ""
        self.weights = []
        self.enters = enters
        self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    def relu(self, x):
        return np.where(x < 0, 0, x)

    def relu_deriv(self, x):
        return np.where(x <= 0, 0, 1)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, x):
        return 1 - (x ** 2)

    def softmax(self, x):
        temp = np.exp(x)
        return temp / np.sum(temp)

    def add_layer(self, neurons, min=-1, max=1):
        if len(self.weights) == 0:
            self.weights.append(np.random.uniform(min, max, (neurons, self.enters)))
        else:
            self.weights.append(
                np.random.uniform(min, max, (neurons, self.weights[len(self.weights) - 1].shape[0])))

    def resize_array(self, x):
        return x.reshape((784, 1))

    def predict(self, input):
        value = np.matmul(self.weights[0], input)
        value = self.relu(value)
        for i in range(1, len(self.weights)):
            value = np.dot(self.weights[i], value)
        return value

    def predict_fun(self, input):
        value = np.matmul(self.weights[0], input)
        value = self.tanh(value)
        for i in range(1, len(self.weights)):
            value = np.dot(self.weights[i], value)
        return value

    def generate_ones(self, x):
        for i in range(int(x.shape[0] / 2)):
            x[i] = 1
        return x

    def fit(self, x, expected_output):
        for i in range(x.shape[0]):
            res_hidden = self.relu(np.dot(self.weights[0], x[i]))
            dropout_mask = np.zeros(res_hidden.shape)
            dropout_mask = self.generate_ones(dropout_mask)
            dropout_mask = shuffle(dropout_mask)
            res_hidden = np.multiply(res_hidden, dropout_mask)
            res_hidden = np.dot(res_hidden, 2)

            res_out = self.weights[1] @ res_hidden

            delta_out = np.dot((2 / res_out.shape[0]), np.subtract(res_out, expected_output[i]))

            delta_hidden = np.multiply(self.weights[1].T @ delta_out, self.relu_deriv(res_hidden))
            delta_hidden = np.multiply(delta_hidden, dropout_mask)

            weight_out = delta_out @ res_hidden.T
            weight_hidden = delta_hidden @ (x[i][:, 0].T.reshape(1, 784))

            self.weights[0] = self.weights[0] - 0.005 * weight_hidden
            self.weights[1] = self.weights[1] - 0.005 * weight_out

    def fit_batch(self, x, expected_output, batch_size, iterations):
        for i in range(iterations):
            how_many = 0
            for j in range(int(x.shape[0] / batch_size)):
                batch_start, batch_stop = ((j * batch_size), ((j + 1) * batch_size))
                batch = x[batch_start:batch_stop].T
                batch_label = np.array(expected_output[batch_start:batch_stop]).reshape((100, 10))
                layer_hidden_1 = self.relu(np.dot(self.weights[0], batch))
                dropout_mask = np.zeros(layer_hidden_1.shape)
                dropout_mask = self.generate_ones(dropout_mask)
                dropout_mask = shuffle(dropout_mask)
                layer_hidden_1 = np.multiply(layer_hidden_1, dropout_mask)
                layer_hidden_1 = np.multiply(layer_hidden_1, 2)
                layer_output = np.dot(self.weights[1], layer_hidden_1)

                layer_output_delta = np.divide(
                    np.dot(2 / layer_hidden_1.shape[0], np.subtract(layer_output, batch_label.T)), batch_size)
                layer_hidden_1_delta = np.dot(self.weights[1].T, layer_output_delta)

                layer_hidden_1_delta = np.multiply(layer_hidden_1_delta, self.relu_deriv(layer_hidden_1))
                layer_hidden_1_delta = np.multiply(layer_hidden_1_delta, dropout_mask)

                layer_output_weight_delta = np.dot(layer_output_delta, layer_hidden_1)
                layer_hidden_1_weight_delta = np.dot(layer_hidden_1_delta, batch.T)

                self.weights[0] = np.subtract(self.weights[0], np.dot(0.1, layer_hidden_1_weight_delta))
                self.weights[1] = np.subtract(self.weights[1], np.dot(0.1, layer_output_weight_delta))

    def fit_batch_fun(self, x, expected_output, batch_size, iterations):
        vec_tanh = np.vectorize(self.tanh)
        vec_tanh_derive = np.vectorize(self.tanh_deriv)

        for i in range(iterations):
            for j in range(int(x.shape[0] / batch_size)):
                batch_start, batch_stop = ((j * batch_size), ((j + 1) * batch_size))
                batch = x[batch_start:batch_stop].T
                batch_label = np.array(expected_output[batch_start:batch_stop]).reshape((100, 10))
                layer_hidden_1 = vec_tanh(np.dot(self.weights[0], batch))

                dropout_mask = np.zeros(layer_hidden_1.shape)
                dropout_mask = self.generate_ones(dropout_mask)
                dropout_mask = shuffle(dropout_mask)
                layer_hidden_1 = np.multiply(layer_hidden_1, dropout_mask)
                layer_hidden_1 = np.multiply(layer_hidden_1, 2)
                layer_output = np.asmatrix(np.dot(self.weights[1], layer_hidden_1))

                for i in range(batch_size):
                    layer_output[:, i] = self.softmax(layer_output[:, i].reshape((10, 1)))

                layer_output = np.asarray(layer_output)
                layer_output_delta = np.divide(
                    np.dot(2 / layer_hidden_1.shape[0], np.subtract(layer_output, batch_label.T)), batch_size)
                layer_hidden_1_delta = np.dot(self.weights[1].T, layer_output_delta)

                layer_hidden_1_delta = np.multiply(layer_hidden_1_delta, vec_tanh_derive(layer_hidden_1))
                layer_hidden_1_delta = np.multiply(layer_hidden_1_delta, dropout_mask)

                layer_output_weight_delta = np.dot(layer_output_delta, layer_hidden_1)
                layer_hidden_1_weight_delta = np.dot(layer_hidden_1_delta, batch.T)

                self.weights[0] = np.subtract(self.weights[0], np.dot(0.2, layer_hidden_1_weight_delta))
                self.weights[1] = np.subtract(self.weights[1], np.dot(0.2, layer_output_weight_delta))

    def save_weights(self, dir):

        if not os.path.exists(dir):
            path = os.path.join(dir)
            os.mkdir(path)

        how_many = 1
        for i in self.weights:
            np.save(f"{dir}/{how_many}", i)
            how_many += 1

    def load_weights(self, dir, size):
        self.weights.clear()
        self.filepath = ""
        for i in range(1, size + 1):
            temp_array = np.asmatrix(np.load(f"{dir}/{i}.npy"))
            self.weights.append(temp_array)

    # def fit_batch(self, x, expected_output, batch_size, iterations):
    #     for i in range(iterations):
    #         for j in range(int(x.shape[0] / batch_size)):
    #             batch_start, batch_stop = ((j * batch_size), ((j + 1) * batch_size))
    #             batch = x[batch_start:batch_stop].reshape(100, 784).T
    #             batch_label = np.array(expected_output[batch_start:batch_stop]).reshape((100, 10))
    #             layer_hidden_1 = self.relu(np.dot(self.weights[0], batch))
    #             dropout_mask = np.zeros(layer_hidden_1.shape)
    #             dropout_mask = self.generate_ones(dropout_mask)
    #             dropout_mask = shuffle(dropout_mask)
    #             layer_hidden_1 = np.multiply(layer_hidden_1, dropout_mask)
    #             layer_hidden_1 = np.multiply(layer_hidden_1, 2)
    #             layer_output = np.dot(self.weights[1], layer_hidden_1)
    #
    #             layer_output_delta = np.divide(
    #                 np.dot(2 / layer_hidden_1.shape[0], np.subtract(layer_output, batch_label.T)), batch_size)
    #             layer_hidden_1_delta = np.dot(self.weights[1].T, layer_output_delta)
    #
    #             layer_hidden_1_delta = np.multiply(layer_hidden_1_delta, self.relu_deriv(layer_hidden_1))
    #             layer_hidden_1_delta = np.multiply(layer_hidden_1_delta, dropout_mask)
    #
    #             layer_output_weight_delta = np.dot(layer_output_delta, layer_hidden_1)
    #             layer_hidden_1_weight_delta = np.dot(layer_hidden_1_delta, batch.T)
    #
    #             self.weights[0] = np.subtract(self.weights[0], np.dot(0.1, layer_hidden_1_weight_delta))
    #             self.weights[1] = np.subtract(self.weights[1], np.dot(0.1, layer_output_weight_delta))
