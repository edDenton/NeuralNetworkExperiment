"""

@author: Edward Denton
"""

import matplotlib.pyplot as plt


class AccuracyPlotter:
    def __init__(self, learn_rate: float, epochs: int, layers: [int], batch_size: int):
        self.epochNumbers = []
        self.training_accuracy = []
        self.test_accuracy = []
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.layers = layers
        self.batch_size = batch_size
        self.figureSize = (10, 7)
        self.plotTitle = "Neural Network Accuracy"
        self.xLabel = f"Epochs ({self.epochs}) with Batch Size of {self.batch_size}"
        self.yLabel = "Accuracy"

    def appendTrainingData(self, epoch: int, accuracy: float):
        self.epochNumbers.append(epoch)
        self.training_accuracy.append(accuracy)

    def appendTestData(self, epoch: int, accuracy: float):
        self.test_accuracy.append(accuracy)

    def averageTestData(self, window):
        averaged, epochs, = [], []

        for i in range(0, len(self.epochNumbers), window):
            chunk = self.test_accuracy[i:i + window]

            if len(chunk) < window:
                break

            averaged.append(sum(chunk) / window)
            epochs.append(self.epochNumbers[i])

        return averaged, epochs

    def showPlot(self):
        test_averaged, test_epochs = self.averageTestData(5)
        print(f"Training Accuracy: {self.training_accuracy[-1] * 100:.2f}%")
        print(f"Test Accuracy: {self.test_accuracy[-1] * 100:.2f}%")
        print(f"Epochs: {self.epochs}, LR: {self.learn_rate}, Layers: {self.layers}, Batch Size: {self.batch_size}")
        plt.figure(figsize=self.figureSize, layout="constrained")
        plt.plot(self.epochNumbers, self.training_accuracy, color="blue", label="Training Accuracy")
        plt.plot(test_epochs, test_averaged, color="red", label="Test Accuracy")
        plt.xlabel(self.xLabel)
        plt.ylabel(self.yLabel)
        plt.title(
            f"Layers: {self.layers}, LR: {self.learn_rate}, TrainAcc: {self.training_accuracy[-1] * 100:.2f}%, TestAcc: {self.test_accuracy[-1] * 100:.2f}%")
        plt.legend(loc="lower right")
        plt.show()
