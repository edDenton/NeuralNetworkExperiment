"""


@author: Edward Denton
"""
import numpy as np
import string
from AccuracyPlotter import AccuracyPlotter


class NeuralNetwork:
    def __init__(self, layer_sizes: [int], learn_rate: float, epochs: int, accuracy_plotter: AccuracyPlotter):
        self.LAYERS = []
        self.EPOCHS = epochs
        self.LEARNING_RATE = learn_rate
        self.NUMOUTPUTS = layer_sizes[-1]

        keys = string.digits + string.ascii_uppercase + string.ascii_lowercase
        self.outputToIndex = {key: index for index, key in enumerate(keys)}
        self.plotter = accuracy_plotter

        for i in range(len(layer_sizes) - 1):
            is_output_layer = (i == len(layer_sizes) - 2)
            self.LAYERS.append(Layer(layer_sizes[i], layer_sizes[i + 1], is_output_layer))

    def oneHotEncoding(self, training_labels: np.array):
        oneHotArray = np.zeros((self.NUMOUTPUTS, len(training_labels)))
        for i, label in enumerate(training_labels):
            oneHotArray[self.outputToIndex[str(label)], i] = 1
        return oneHotArray

    def forward_propagation(self, inputs: np.array):
        for layer in self.LAYERS:
            inputs = layer.calculateOutputs(inputs)
        return inputs

    def back_propagation(self, outputs: np.array, training_labels: np.array):
        batch_size = len(training_labels)
        one_hot_labels = self.oneHotEncoding(training_labels)
        gradient = outputs - one_hot_labels

        self.LAYERS[-1].calculateGradients(gradient, batch_size)

        for i in range(len(self.LAYERS) - 2, -1, -1):
            gradient = np.dot(self.LAYERS[i + 1].weights.T, gradient) * self.LAYERS[i].ReLUDerivative(
                self.LAYERS[i].layerNodeInfo.preActivationValues
            )
            self.LAYERS[i].calculateGradients(gradient, batch_size)

    def updateWeightsBiases(self):
        for layer in self.LAYERS:
            layer.updateWeightsBiases(self.LEARNING_RATE)

    def prediction_accuracy(self, outputs: np.array, training_labels: np.array):
        predictions = np.argmax(outputs, axis=0)
        return np.sum(predictions == training_labels) / len(training_labels)

    def train(self, training_images: np.array, training_labels: np.array, batch_size: int):
        num_samples = len(training_labels)
        for epoch in range(self.EPOCHS):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            training_images = training_images[:, indices]
            training_labels = training_labels[indices]
            training_accuracy = []

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_images = training_images[:, start:end]
                batch_labels = training_labels[start:end]
                outputs = self.forward_propagation(batch_images)

                training_accuracy.append(self.prediction_accuracy(outputs, batch_labels))

                self.back_propagation(outputs, batch_labels)
                self.updateWeightsBiases()

            self.plotter.appendTrainingData(epoch=epoch,
                                            accuracy=(sum(training_accuracy) / len(training_accuracy)))

    def test(self, test_images: np.array, test_labels: np.array):
        outputs = self.forward_propagation(test_images)
        self.plotter.setTestAccuracy(self.prediction_accuracy(outputs, test_labels))

    def makePrediction(self, images: np.array):
        outputs = self.forward_propagation(images)
        return outputs


class LayerNodeInfo:
    def __init__(self):
        self.nodeValues = np.array([])
        self.preActivationValues = np.array([])
        self.activationValues = np.array([])


class Layer:
    def __init__(self, numNodesIn: int, numNodesOut: int, outputLayer: bool):
        self.numNodesIn = numNodesIn
        self.numNodesOut = numNodesOut
        self.outputLayer = outputLayer

        self.layerNodeInfo = LayerNodeInfo()

        self.weights = np.random.randn(numNodesOut, numNodesIn) * np.sqrt(2 / numNodesIn)
        self.biases = np.zeros((numNodesOut, 1))

        self.costGradientWeights = np.zeros((numNodesOut, numNodesIn))
        self.costGradientBiases = np.zeros((numNodesOut, 1))

    def calculateOutputs(self, inputs: np.array):
        self.layerNodeInfo.nodeValues = inputs
        self.layerNodeInfo.preActivationValues = np.dot(self.weights, inputs) + self.biases

        if self.outputLayer:
            self.layerNodeInfo.activationValues = self.softmax(self.layerNodeInfo.preActivationValues)
        else:
            self.layerNodeInfo.activationValues = self.ReLU(self.layerNodeInfo.preActivationValues)

        return self.layerNodeInfo.activationValues

    def calculateGradients(self, outputs: np.array, batch_size: int):
        self.costGradientWeights = (1 / batch_size) * np.dot(outputs, self.layerNodeInfo.nodeValues.T)
        self.costGradientBiases = (1 / batch_size) * np.sum(outputs, axis=1, keepdims=True)

    def updateWeightsBiases(self, learn_rate: float):
        self.weights -= learn_rate * self.costGradientWeights
        self.biases -= learn_rate * self.costGradientBiases

    def ReLU(self, inputs: np.array):
        return np.maximum(0, inputs)

    def ReLUDerivative(self, inputs: np.array):
        return (inputs > 0).astype(float)

    def softmax(self, inputs: np.array):
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))  # Stable softmax
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)
