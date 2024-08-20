"""

@author: Edward Denton
"""

import matplotlib.pyplot as plt

class AccuracyPlotter:
    def __init__(self, learn_rate: float, epochs: int, layers: [int]):
        self.epochNumbers = []
        self.training_accuracy = []
        self.test_accuracy = []
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.layers = layers
        self.figureSize = (10, 7)
        self.plotTitle = "Neural Network Accuracy"
        self.xLabel = "Epochs"
        self.yLabel = "Accuracy"

    def appendData(self, epoch: int, accuracy: float):
        self.epochNumbers.append(epoch)
        self.training_accuracy.append(accuracy)

    def setTestAccuracy(self, accuracy: float):
        self.test_accuracy = [accuracy for i in range(len(self.epochNumbers))]

    def showPlot(self):
        plt.figure(figsize=self.figureSize, layout="constrained")
        plt.plot(self.epochNumbers, self.training_accuracy, color="blue", label="Training Accuracy")
        plt.plot(self.epochNumbers, self.test_accuracy, color="red", label="Test Accuracy")
        plt.xlabel(self.xLabel)
        plt.ylabel(self.yLabel)
        plt.title(f"Layers: {self.layers}, LR: {self.learn_rate}, Epochs: {self.epochs}, Training Accuracy: {self.training_accuracy[-1]*100:.2f}%, Test Accuracy: {self.test_accuracy[-1]*100:.2f}%")
        plt.legend(loc="lower right")
        plt.show()
