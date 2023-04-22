import os

import numpy as np
from sklearn import preprocessing


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

    def fit(self, x, expected_output):
        for i in range(x.shape[0]):
            res_hidden = np.dot(self.weights[0], x[i])
            res_hidden = self.relu(res_hidden)

            res_out = self.weights[1] @ res_hidden

            delta_out = np.dot((2 / res_out.shape[0]), np.subtract(res_out, expected_output[i]))

            delta_hidden = np.multiply(self.weights[1].T @ delta_out, self.relu_deriv(res_hidden))

            weight_out = delta_out @ res_hidden.T
            weight_hidden = delta_hidden @ (x[i][:, 0].T.reshape(1, 784))

            self.weights[0] = self.weights[0] - 0.01 * weight_hidden
            self.weights[1] = self.weights[1] - 0.01 * weight_out

    def neural_network_learn(self, x, y, alpha):
        relu_vector = np.vectorize(self.relu)
        relu_deriv_vector = np.vectorize(self.relu_deriv)

        for i in range(y.shape[1]):
            res_hidden = np.dot(self.weights[0], x[:, i])
            res_hidden = self.relu(res_hidden)

            # res_out = self.weights[1] * res_hidden
            res_out = np.dot(self.weights[1], res_hidden)

            # delta_out =  * (res_out - y[:, i])
            delta_out = np.dot((2 / res_out.shape[0]), np.subtract(res_out, y[:, i]))

            delta_hidden = np.multiply(self.weights[1].T * delta_out, relu_deriv_vector(res_hidden))

            weight_out = delta_out * res_hidden.T
            weight_hidden = delta_hidden * x[:, i].T

            self.weights[0] = self.weights[0] - alpha * weight_hidden
            self.weights[1] = self.weights[1] - alpha * weight_out

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
