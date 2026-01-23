"""

@author: Edward Denton
"""
import numpy as np
import string
from AccuracyPlotter import AccuracyPlotter


class NeuralNetwork:
    def __init__(self, layers, learn_rate: float, epochs: int, accuracy_plotter: AccuracyPlotter):
        self.LAYERS = layers
        self.EPOCHS = epochs
        self.LEARNING_RATE = learn_rate
        self.NUMOUTPUTS = layers[-1].numNodesOut

        keys = string.digits + string.ascii_uppercase + string.ascii_lowercase
        self.outputToIndex = {key: index for index, key in enumerate(keys)}
        self.plotter = accuracy_plotter

    def oneHotEncoding(self, training_labels: np.array):
        oneHotArray = np.zeros((self.NUMOUTPUTS, len(training_labels)))
        for i, label in enumerate(training_labels):
            oneHotArray[self.outputToIndex[str(label)], i] = 1
        return oneHotArray

    def forward_propagation(self, inputs: np.array):
        for layer in self.LAYERS:
            inputs = layer.forward_propagation(inputs)
        return inputs

    def back_propagation(self, outputs: np.array, training_labels: np.array):
        batch_size = len(training_labels)
        one_hot_labels = self.oneHotEncoding(training_labels)
        gradient = outputs - one_hot_labels

        for layer in reversed(self.LAYERS):
            gradient = layer.back_propagation(gradient, batch_size)

    def updateWeightsBiases(self):
        for layer in self.LAYERS:
            layer.updateWeightsBiases(self.LEARNING_RATE)

    def prediction_accuracy(self, outputs: np.array, training_labels: np.array):
        predictions = np.argmax(outputs, axis=0)
        return np.sum(predictions == training_labels) / len(training_labels)

    def train(self, training_images: np.array, training_labels: np.array, batch_size: int, test_images, test_labels):
        num_samples = len(training_labels)
        for epoch in range(self.EPOCHS):
            if epoch % 50 == 0:
                print("Epoch Completed: #" + str(epoch))

            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            training_images = training_images[indices]
            training_labels = training_labels[indices]
            training_accuracy = []

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_images = training_images[start:end]
                batch_labels = training_labels[start:end]
                outputs = self.forward_propagation(batch_images)

                training_accuracy.append(self.prediction_accuracy(outputs, batch_labels))

                self.back_propagation(outputs, batch_labels)
                self.updateWeightsBiases()

            self.plotter.appendTrainingData(epoch=epoch,
                                            accuracy=(sum(training_accuracy) / len(training_accuracy)))
            self.test(test_images, test_labels, epoch)

    def test(self, test_images: np.array, test_labels: np.array, epoch: int):
        outputs = self.forward_propagation(test_images)
        self.plotter.appendTestData(epoch=epoch,
                                    accuracy=self.prediction_accuracy(outputs, test_labels))

    def makePrediction(self, images: np.array):
        outputs = self.forward_propagation(images)
        return outputs


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
            2.0 / (in_channels * kernel_size * kernel_size))
        self.biases = np.zeros((out_channels, 1))

        self.image_shape = None
        self.input_col = None
        self.preactivationValues = None
        self.costGradientWeights = None
        self.costGradientBiases = None

        self.H_output = None
        self.W_output = None

    def im2col(self, inputs):
        batch, C_in, H_in, W_in = self.image_shape

        X_pad = np.pad(inputs, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        X_col = np.zeros((batch, C_in * self.kernel_size * self.kernel_size, self.H_output * self.W_output))

        col_idx = 0
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                patch = X_pad[:, :, i: i + self.stride * self.H_output: self.stride,
                        j: j + self.stride * self.W_output: self.stride]
                X_col[:, col_idx * C_in:(col_idx + 1) * C_in, :] = patch.reshape(batch, C_in, -1)
                col_idx += 1

        return X_col

    def col2im(self, outputs):
        batch, C_in, H_in, W_in = self.image_shape
        dX_pad = np.zeros((batch, C_in, H_in + 2 * self.padding, W_in + 2 * self.padding))

        col_idx = 0
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                gradient = outputs[:, col_idx * C_in:(col_idx + 1) * C_in, :].reshape(batch, C_in, self.H_output,
                                                                                      self.W_output)
                dX_pad[:, :, i: i + self.stride * self.H_output: self.stride,
                j: j + self.stride * self.W_output: self.stride] += gradient
                col_idx += 1

        dX = dX_pad[:, :, self.padding:self.padding + H_in, self.padding:self.padding + W_in]

        return dX

    def forward_propagation(self, inputs):
        self.image_shape = inputs.shape
        batch, C_in, H_in, W_in = self.image_shape
        self.H_output = int(np.floor((H_in + 2 * self.padding - self.kernel_size) / self.stride) + 1)
        self.W_output = int(np.floor((W_in + 2 * self.padding - self.kernel_size) / self.stride) + 1)

        X_col = self.im2col(inputs)
        self.input_col = X_col
        W_col = np.reshape(self.weights, (self.out_channels, C_in * self.kernel_size * self.kernel_size))
        X_col_reshaped = X_col.transpose(1, 0, 2).reshape(
            W_col.shape[1], -1
        )
        Y_col = W_col @ X_col_reshaped + self.biases
        Y_col = Y_col.reshape(self.out_channels, batch, self.H_output * self.W_output).transpose(1, 0, 2)
        self.preactivationValues = Y_col
        A_col = self.ReLU(Y_col)
        A = np.reshape(A_col, (batch, self.out_channels, self.H_output, self.W_output))

        return A

    def back_propagation(self, gradient, batch_size):
        batch, C_in, H_in, W_in = self.image_shape
        dA_col = np.reshape(gradient, (batch, self.out_channels, self.W_output * self.H_output))

        dY_col = dA_col * self.ReLUDerivative(self.preactivationValues)

        self.calculateGradients(dY_col, batch_size)

        W_col = np.reshape(self.weights, (self.out_channels, C_in * self.kernel_size * self.kernel_size))
        dY_col_reshaped = dY_col.transpose(1, 0, 2).reshape(self.out_channels, -1)
        dX_col = W_col.T @ dY_col_reshaped
        dX_col = dX_col.reshape(W_col.shape[1], batch, self.H_output * self.W_output).transpose(1, 0, 2)
        dX = self.col2im(dX_col)

        return dX

    def calculateGradients(self, gradients, batch_size):
        dW_col = np.zeros((self.out_channels, self.in_channels * self.kernel_size * self.kernel_size))
        for b in range(batch_size):
            dW_col += gradients[b] @ np.transpose(self.input_col[b])
        self.costGradientWeights = (1 / batch_size) * np.reshape(dW_col, self.weights.shape)
        self.costGradientBiases = np.sum(gradients, axis=(0, 2), keepdims=False)
        self.costGradientBiases = (1 / batch_size) * np.reshape(self.costGradientBiases, (self.out_channels, 1))

    def updateWeightsBiases(self, learn_rate):
        self.weights = self.weights - learn_rate * self.costGradientWeights
        self.biases = self.biases - learn_rate * self.costGradientBiases

    def ReLU(self, inputs):
        return np.maximum(0, inputs)

    def ReLUDerivative(self, inputs):
        return (inputs > 0).astype(float)

    def __str__(self):
        return f"Conv2D({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding})"

    def __repr__(self):
        return f"Conv2D({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding})"


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.mask = None

    def forward_propagation(self, inputs):
        self.input = inputs
        batch, C, H, W = inputs.shape

        assert H % self.pool_size == 0
        assert W % self.pool_size == 0

        H_out = H // self.pool_size
        W_out = W // self.pool_size

        # Reshape so each pooling window is explicit
        x = inputs.reshape(
            batch,
            C,
            H_out,
            self.pool_size,
            W_out,
            self.pool_size
        )

        # Max over pooling dimensions
        out = x.max(axis=(3, 5))

        # Save mask for backprop
        self.mask = (x == out[:, :, :, None, :, None])

        return out

    def back_propagation(self, gradient, batch_size):
        batch, C, H, W = self.input.shape
        H_out = H // self.pool_size
        W_out = W // self.pool_size

        # Reshape gradient to broadcast over pooling window
        grad_expanded = gradient[:, :, :, None, :, None]

        # Distribute gradient to max locations
        dX = self.mask * grad_expanded

        return dX.reshape(self.input.shape)

    def updateWeightsBiases(self, learn_rate):
        pass

    def __str__(self):
        return f"MaxPool2D({self.pool_size}, {self.stride})"

    def __str__(self):
        return f"MaxPool2D({self.pool_size}, {self.stride})"

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward_propagation(self, inputs):
        self.input_shape = inputs.shape
        batch, C, H, W = self.input_shape
        inputs = np.reshape(inputs, (batch, C * H * W))
        inputs = np.transpose(inputs)
        return inputs

    def back_propagation(self, gradient, batch_size):
        return np.reshape(gradient, self.input_shape)

    def updateWeightsBiases(self, learn_rate):
        pass

    def __str__(self):
        return f"Flatten"

    def __repr__(self):
        return f"Flatten"


class DenseLayerNodeInfo:
    def __init__(self):
        self.nodeValues = np.array([])
        self.preActivationValues = np.array([])
        self.activationValues = np.array([])


class DenseLayer:
    def __init__(self, numNodesIn: int, numNodesOut: int, outputLayer: bool):
        self.numNodesIn = numNodesIn
        self.numNodesOut = numNodesOut
        self.outputLayer = outputLayer

        self.layerNodeInfo = DenseLayerNodeInfo()

        self.weights = np.random.randn(numNodesOut, numNodesIn) * np.sqrt(2 / numNodesIn)
        self.biases = np.zeros((numNodesOut, 1))

        self.costGradientWeights = np.zeros((numNodesOut, numNodesIn))
        self.costGradientBiases = np.zeros((numNodesOut, 1))

    def forward_propagation(self, inputs: np.array):
        self.layerNodeInfo.nodeValues = inputs
        self.layerNodeInfo.preActivationValues = np.dot(self.weights, inputs) + self.biases

        if self.outputLayer:
            self.layerNodeInfo.activationValues = self.softmax(self.layerNodeInfo.preActivationValues)
        else:
            self.layerNodeInfo.activationValues = self.ReLU(self.layerNodeInfo.preActivationValues)

        return self.layerNodeInfo.activationValues

    def back_propagation(self, gradient, batch_size):
        if not self.outputLayer:
            dY = gradient * self.ReLUDerivative(self.layerNodeInfo.preActivationValues)
        else:
            dY = gradient

        self.calculateGradients(dY, batch_size)
        dX = np.transpose(self.weights) @ dY

        return dX

    def calculateGradients(self, gradient: np.array, batch_size: int):
        self.costGradientWeights = (1 / batch_size) * (gradient @ self.layerNodeInfo.nodeValues.T)
        self.costGradientBiases = (1 / batch_size) * np.sum(gradient, axis=1, keepdims=True)

    def updateWeightsBiases(self, learn_rate: float):
        self.weights -= learn_rate * self.costGradientWeights
        self.biases -= learn_rate * self.costGradientBiases

    def ReLU(self, inputs: np.array):
        return np.maximum(0, inputs)

    def ReLUDerivative(self, inputs: np.array):
        return (inputs > 0).astype(float)

    def softmax(self, inputs: np.array):
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)

    def __str__(self):
        return f"Dense({self.numNodesIn}, {self.numNodesOut})"

    def __repr__(self):
        return f"Dense({self.numNodesIn}, {self.numNodesOut})"
