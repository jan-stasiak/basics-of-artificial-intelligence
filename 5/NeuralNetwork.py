import os

import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle


class NeuralNetwork:
    def __init__(self, enters=0):
        self.filepath = ""
        self.weights = []
        self.filters = []
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

    def add_filters(self, min=-0.1, max=0.1, size_x=3, size_y=3, how_many=1):
        out_put = [np.random.uniform(min, max, (size_x, size_y)) for i in range(how_many)]
        return out_put

    def resize_array(self, x):
        return x.reshape((784, 1))

    def predict(self, input):
        value = np.matmul(self.weights[0], input)
        value = self.relu(value)
        for i in range(1, len(self.weights)):
            value = np.dot(self.weights[i], value)
        return value

    def predict_zad2(self, image, kernels, label):
        kernel_layer = np.zeros((676, 16))
        image_sections = 0
        for j in range(16):
            res, image_sections = self.conv_zad1(image, kernels[j])
            kernel_layer[:, j] = res[:, 0]
        kernel_layer_flatten = self.relu(kernel_layer.reshape((1, 10816)))
        layer_output = self.weights[0] @ kernel_layer_flatten.T
        if np.argmax(layer_output) == np.argmax(label.reshape((10, 1))):
            return 1
        return 0

    def predict_zad3(self, image, kernels, label):
        kernel_layer = np.zeros((676, 16))
        image_sections = 0
        for j in range(16):
            res, image_sections = self.conv_zad1(image, kernels[j])
            kernel_layer[:, j] = res[:, 0]
        index = []
        kernel_layer_pooling = np.zeros((169, 16))
        for j in range(16):
            res, temp_index = self.pooling(kernel_layer[:, j].reshape((26, 26)), 2, 2)
            index.append(temp_index)
            kernel_layer_pooling[:, j] = res.reshape((169))
        kernel_layer_flatten = self.relu(kernel_layer_pooling.reshape((2704, 1)))

        layer_output = self.weights[0] @ kernel_layer_flatten
        if np.argmax(layer_output) == np.argmax(label.reshape((10, 1))):
            return 1
        return 0

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

    def conv_zad1(self, input_image, filter):
        (size_x, size_y) = input_image.shape
        (filter_x, filter_y) = filter.shape
        output_img = np.zeros(((size_y - filter_y + 1) * (size_x - filter_x + 1), (filter_x * filter_y)))
        res = np.zeros((1, filter_x * filter_y))
        filter = filter.reshape((1, 9))
        k = 0
        for i in range(0, size_x - filter_x + 1, 1):
            for j in range(0, size_y - filter_y + 1, 1):
                res = input_image[i:i + 3, j:j + 3].reshape(1, 9)
                output_img[k] = res
                k += 1
        temp = output_img @ filter.T
        return temp, output_img


    def fit_zad2(self, image, label, kernels, alpha, iteration):
        for z in range(iteration):
            how_many = 0
            for i in range(image.shape[0]):
                kernel_layer = np.zeros((676, 16))
                image_sections = 0
                for j in range(16):
                    res, image_sections = self.conv_zad1(image[i], kernels[j])
                    kernel_layer[:, j] = res[:, 0]
                kernel_layer_flatten = self.relu(kernel_layer.reshape((1, 10816)))
                layer_output = self.weights[0] @ kernel_layer_flatten.T
                if np.argmax(layer_output) == np.argmax(label[i].reshape((10, 1))):
                    how_many = how_many + 1
                layer_output_delta = np.dot((2 / layer_output.shape[0]),
                                            np.subtract(layer_output, label[i].reshape((10, 1))))
                kernel_layer_1_delta = self.weights[0].T @ layer_output_delta
                kernel_layer_1_delta = np.multiply(kernel_layer_1_delta, self.relu_deriv(kernel_layer_flatten).T)
                kernel_layer_1_delta_reshaped = kernel_layer_1_delta.reshape((676, 16))
                layer_output_weight_delta = layer_output_delta @ kernel_layer_flatten
                kernel_layer_1_weight_delta = np.dot(kernel_layer_1_delta_reshaped.T, image_sections)
                self.weights[0] = np.subtract(self.weights[0], np.dot(alpha, layer_output_weight_delta))
                for j in range(16):
                    kernels[j] = np.subtract(kernels[j],
                                             np.dot(alpha, kernel_layer_1_weight_delta[j, :].reshape((3, 3))))
            print(f"Skutecznosc sieci - train {z + 1}: {how_many / 60_000:.4%}")

    def pooling(self, image, mask, size):
        res = np.zeros((int(image.shape[0] / mask), int(image.shape[1] / mask)))
        image_index = np.zeros(image.shape)
        row, col = 0, 0
        for i in range(0, image.shape[0], size):
            for j in range(0, image.shape[1], size):
                temp = image[i:i + 2, j:j + 2]
                (x, y) = np.unravel_index(np.argmax(image[i:i + size, j:j + size], axis=None), image[i:i + size, j:j + size].shape)
                image_index[i + x, j + y] = 1
                res[col, row] = temp[x, y]
                row += 1
            col += 1
            row = 0

        return res, image_index

    def fit_zad3(self, image, label, kernels, alpha, iteration, all_possible):
        for z in range(iteration):
            how_many = 0
            for i in range(image.shape[0]):
                kernel_layer = np.zeros((676, 16))
                image_sections = 0
                for j in range(16):
                    res, image_sections = self.conv_zad1(image[i], kernels[j])
                    kernel_layer[:, j] = res[:, 0]
                index = []
                kernel_layer_pooling = np.zeros((169, 16))
                for j in range(16):
                    res, temp_index = self.pooling(kernel_layer[:,j].reshape((26,26)), 2, 2)
                    index.append(temp_index)
                    kernel_layer_pooling[:, j] = res.reshape((169))
                kernel_layer_flatten = self.relu(kernel_layer_pooling.reshape((2704, 1)))

                layer_output = self.weights[0] @ kernel_layer_flatten
                if np.argmax(layer_output) == np.argmax(label[i].reshape((10, 1))):
                    how_many = how_many + 1
                layer_output_delta = np.dot((2 / layer_output.shape[0]),
                                            np.subtract(layer_output, label[i].reshape((10, 1))))
                kernel_layer_1_delta = self.weights[0].T @ layer_output_delta
                kernel_layer_1_delta = np.multiply(kernel_layer_1_delta, self.relu_deriv(kernel_layer_flatten))
                kernel_layer_1_delta_reshaped = kernel_layer_1_delta.reshape((169, 16))
                layer_output_weight_delta = layer_output_delta @ kernel_layer_flatten.T

                kernel_layer_1_delta_reshaped_big = np.zeros((676, 16))
                for j in range(16):
                    col, row = 0, 0
                    kernel_layer_1_delta_reshaped[:, j].reshape(13, 13)
                    temp = np.zeros((26,26))
                    for a in range(0,25,2):
                        for b in range(0,25,2):
                            temp[a:a+2,b:b+2] = kernel_layer_1_delta_reshaped[:, j].reshape(13, 13)[col, row]
                            row += 1
                        row = 0
                        col += 1
                    kernel_layer_1_delta_reshaped_big[:, j] = np.multiply(temp, index[0]).reshape((676))


                kernel_layer_1_weight_delta = np.dot(kernel_layer_1_delta_reshaped_big.T, image_sections)
                self.weights[0] = np.subtract(self.weights[0], np.dot(alpha, layer_output_weight_delta))
                for j in range(16):
                    kernels[j] = np.subtract(kernels[j],
                                                     np.dot(alpha, kernel_layer_1_weight_delta[j, :].reshape((3, 3))))
            print(f"Skutecznosc sieci - test {z + 1}: {how_many / all_possible:.4%}")


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

    def save_weights(self, dir, kernel):

        if not os.path.exists(dir):
            path = os.path.join(dir)
            os.mkdir(path)

        how_many = 1
        for i in self.weights:
            np.save(f"{dir}/{how_many}", i)
            how_many += 1

        how_many = 1
        for i in kernel:
            np.save(f"{dir}/k{how_many}", i)
            how_many += 1

    def load_weights(self, dir, size):
        self.weights.clear()
        self.filepath = ""
        for i in range(1, size + 1):
            temp_array = np.asmatrix(np.load(f"{dir}/{i}.npy"))
            self.weights.append(temp_array)
