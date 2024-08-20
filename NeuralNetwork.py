"""


@author: Edward Denton
"""
import numpy as np
from AccuracyPlotter import AccuracyPlotter


class NeuralNetwork:
    def __init__(self, layer_sizes: [int], learn_rate: float, epochs: int, accuracy_plotter: AccuracyPlotter):
        self.LAYERS = []
        self.EPOCHS = epochs
        self.LEARNING_RATE = learn_rate
        self.plotter = accuracy_plotter
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                self.LAYERS.append(Layer(layer_sizes[i], layer_sizes[i + 1], True))
            else:
                self.LAYERS.append(Layer(layer_sizes[i], layer_sizes[i + 1], False))

    def oneHotEncoding(self, training_labels: np.array):
        oneHotArrays = []
        for i in range(len(training_labels)):
            oneHotArray = np.array([0 for _ in range(10)])
            oneHotArray[training_labels[i]] = 1
            oneHotArray.reshape(len(oneHotArray), 1)
            oneHotArrays.append(oneHotArray)
        return np.transpose(np.array(oneHotArrays))

    def forward_propagation(self, inputs: np.array):
        for layer in self.LAYERS:
            inputs = layer.calculateOutputs(inputs)
        return inputs

    def back_propagation(self, outputs: np.array, training_labels: np.array):
        outputs = 2 * (outputs - self.oneHotEncoding(training_labels))
        self.LAYERS[-1].costGradientWeights = (1 / len(training_labels)) * outputs.dot(
            np.transpose(self.LAYERS[-2].layerNodeInfo.activationValues))
        self.LAYERS[-1].calculateGradients(outputs, training_labels)

        for i in range(len(self.LAYERS) - 2, -1, -1):
            outputs = np.transpose(self.LAYERS[i + 1].weights).dot(outputs) * self.LAYERS[i].ReLUDerivative(
                self.LAYERS[i].layerNodeInfo.preActivationValues)
            self.LAYERS[i].calculateGradients(outputs, training_labels)

    def updateWeightsBiases(self, learn_rate: float):
        for layer in self.LAYERS:
            layer.updateWeightsBiases(learn_rate)

    def prediction_accuracy(self, outputs: np.array, training_labels: np.array):
        return np.sum(np.argmax(outputs, 0) == training_labels) / len(training_labels)

    def train(self, training_images: np.array, training_labels: np.array):
        for epoch in range(self.EPOCHS):
            outputs = self.forward_propagation(training_images)

            self.plotter.appendData(epoch=epoch,
                                    accuracy=self.prediction_accuracy(outputs, training_labels))

            self.back_propagation(outputs, training_labels)
            self.updateWeightsBiases(self.LEARNING_RATE)

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

        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(numNodesOut, numNodesIn))
        self.biases = np.zeros((numNodesOut, 1))
        self.costGradientWeights = np.zeros((numNodesOut, numNodesIn))
        self.costGradientBiases = np.zeros((numNodesOut, 1))

    def calculateOutputs(self, inputs: np.array):
        self.layerNodeInfo.nodeValues = inputs
        self.layerNodeInfo.preActivationValues = np.array(np.dot(self.weights, inputs) + self.biases)
        if self.outputLayer:
            self.layerNodeInfo.activationValues = self.softmax(self.layerNodeInfo.preActivationValues)
            return self.layerNodeInfo.activationValues
        else:
            self.layerNodeInfo.activationValues = self.ReLU(self.layerNodeInfo.preActivationValues)
            return self.layerNodeInfo.activationValues

    def calculateGradients(self, outputs: np.array, training_labels: np.array):
        self.costGradientWeights = (1 / len(training_labels)) * outputs.dot(np.transpose(self.layerNodeInfo.nodeValues))
        self.costGradientBiases = (1 / len(training_labels)) * np.sum(outputs)
        NONE = None

    def updateWeightsBiases(self, learn_rate: float):
        self.weights = self.weights - learn_rate * self.costGradientWeights
        self.biases = self.biases - learn_rate * self.costGradientBiases
        NONE = None

    def ReLU(self, inputs: np.array):
        return np.maximum(0, inputs)

    def ReLUDerivative(self, inputs: np.array):
        return inputs > 0

    def sigmoid(self, inputs: np.array):
        return 1 / (1 + np.exp(-inputs))

    def sigmoidDerivative(self, inputs: np.array):
        return inputs * (1 - inputs)

    def softmax(self, inputs: np.array):
        return np.exp(inputs) / np.sum(np.exp(inputs), axis=0)
