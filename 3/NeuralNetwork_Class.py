import os.path

import numpy as np


class NeuralNetwork:

    def __init__(self, enters):
        self.filepath = ""
        self.weights = []
        self.enters = enters

    def add_layer(self, neurons, min=-1, max=1):
        if len(self.weights) == 0:
            self.weights.append(np.asmatrix(np.random.uniform(min, max, (neurons, self.enters))))
        else:
            self.weights.append(
                np.asmatrix(np.random.uniform(min, max, (neurons, self.weights[len(self.weights) - 1].shape[0]))))

    def relu(self, x):
        return np.where(x < 0, 0, x)

    def relu_deriv(self, x):
        return np.where(x <= 0, 0, 1)

    def neural_network_first(self, x):
        relu_vector = np.vectorize(self.relu)
        output = []
        for i in range(x.shape[1]):
            res_hidden = self.weights[0] * x[:, i]
            res_hidden = self.relu(res_hidden)

            res_out = self.weights[1] * res_hidden
            output.append(res_out)

        return output

    def neural_network_learn(self, x, y, alpha):
        relu_vector = np.vectorize(self.relu)
        relu_deriv_vector = np.vectorize(self.relu_deriv)

        output = []
        for i in range(y.shape[1]):
            res_hidden = self.weights[0] * x[:, i]
            res_hidden = self.relu(res_hidden)

            res_out = self.weights[1] * res_hidden
            output.append(res_out)

            delta_out = (2 / res_out.shape[0]) * (res_out - y[:, i])
            delta_hidden = np.multiply(self.weights[1].T * delta_out, relu_deriv_vector(res_hidden))

            weight_out = delta_out * res_hidden.T
            weight_hidden = delta_hidden * x[:, i].T

            self.weights[0] = self.weights[0] - alpha * weight_hidden
            self.weights[1] = self.weights[1] - alpha * weight_out

        return output

    def reload_weights(self):
        self.load_weights(self.filepath)



    def predict(self, input):
        value = np.matmul(self.weights[0], input)
        for i in range(1, len(self.weights)):
            value = np.dot(self.weights[i], value)
        return value

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

    def load_weights_form_txt(self, file_name):
        self.filepath = file_name
        f = open(file_name, "r")
        a = f.readlines()

        network = []
        l = []
        for line in a:
            if line == '#\n' or line == '#':
                network.append(np.matrix(l))
                l.clear()
            else:
                l.append(list(map(float, list(line.replace("\n", "").replace("âˆ’", "-").split(" ")))))
        self.weights = network

    def load_colors(self, file_name, rgb, colors):
        f = open(file_name, "r")
        a = f.readlines()
        for line in a:
            temp = np.array(list(map(float, line.replace("\n", "").split(' '))))
            rgb.append(np.asmatrix(temp[:3].reshape(3, 1)))
            temp_color = np.asmatrix(np.zeros((4, 1)))
            temp_color[int(temp[3]) - 1] = 1
            colors.append(temp_color)
