import numpy as np


class NeuralNetwork:

    def __init__(self, n=0, inp=0, min_max=0):
        self.weights = []
        self.err = 0
        if n != 0:
            if min_max == 0:
                self.weights.append(np.random.uniform(-1, 1, (n, inp)))
            else:
                self.weights.append(np.random.uniform(min_max[0], min_max[1], (n, inp)))

    def neural_network_smart(self, x, y, alpha):
        res = np.asmatrix(np.zeros((y.shape[0], x.shape[1])))
        for i in range(y.shape[1]):
            output = self.weights[0] * x[:, i]
            delta = 2 / self.weights[0].shape[0] * np.outer(output - y[:, i], x[:, i])
            self.weights[0] = self.weights[0] - alpha * delta
            res[:, i] = output

        self.err = (1 / self.weights[0].shape[0]) * np.square(res - y)

    def add_layer(self, n, min_max=0):
        if min_max == 0:
            self.weights.append(np.random.uniform(-1, 1, (n, self.weights[-1].shape[0])))
        else:
            self.weights.append(np.random.uniform(min_max[0], min_max[1], (n, self.weights[-1].shape[0])))

    def predict(self, input):
        value = np.matmul(self.weights[0], input)
        for i in range(1, len(self.weights)):
            value = np.dot(self.weights[i], value)
        return value

    def load_weights(self, file_name):
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
            # print(line)

            B = list(map(float, line.replace("\n", "").split(' ')))
            temp = np.array(B)
            C = temp[:3]
            D = temp[3]

            rgb.append(np.asmatrix(C.reshape(3, 1)))
            if (D == 1):
                temp_color = np.matrix([[1], [0], [0], [0]])
            elif (D == 2):
                temp_color = np.matrix([[0], [1], [0], [0]])
            elif (D == 3):
                temp_color = np.matrix([[0], [0], [1], [0]])
            else:
                temp_color = np.matrix([[0], [0], [0], [1]])

            colors.append(temp_color)
