import numpy as np


class NeuralNetwork:

    def __init__(self, n=0, inp=0, min_max=0):
        self.weights = []
        if n != 0:
            if min_max == 0:
                self.weights.append(np.random.uniform(-1, 1, (n, inp)))
            else:
                self.weights.append(np.random.uniform(min_max[0], min_max[1], (n, inp)))

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
                network.append(np.array(l))
                l.clear()
            else:
                l.append(list(map(float, list(line.replace("\n", "").replace("âˆ’", "-").split(" ")))))
        self.weights = network
