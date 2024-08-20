"""


@author: Edward Denton
"""
import numpy as np
import pandas as pd
import random as rand

from matplotlib import pyplot as plt

from NeuralNetwork import NeuralNetwork
from AccuracyPlotter import AccuracyPlotter


def getData():
    data = pd.read_csv('data.csv')
    data = np.array(data)
    np.random.shuffle(data)

    testing_data = np.transpose(data[0:2000])
    testing_labels = testing_data[0]
    testing_images = testing_data[1:] / 255.0

    training_data = np.transpose(data[2000:])
    training_labels = training_data[0]
    training_images = training_data[1:] / 255.0

    return training_images, training_labels, testing_images, testing_labels


def seePerformance(NN: NeuralNetwork, images, labels):
    userInput = ""
    lastIncorrect = 0
    predictions = NN.makePrediction(images)

    while userInput != "stop":
        if userInput != "":
            if lastIncorrect == len(images):
                lastIncorrect = 0
            for i in range(lastIncorrect, len(images)):
                if np.argmax(predictions[:, i]) != labels[i]:
                    lastIncorrect += 1
                    plt.imshow(images[:, i].reshape(28, 28) * 255.0, cmap='Greys')
                    plt.title(f"Prediction: {np.argmax(predictions[:, i])}, Label: {labels[i]}")
                    plt.show()
                    break
        else:
            index = rand.randint(0, len(images) - 1)
            plt.imshow(images[:, index, None].reshape(28, 28) * 255.0, cmap='Greys')
            plt.title(f"Prediction: {np.argmax(predictions[:, index])}, Label: {labels[index]}")
            plt.show()
        userInput = input()


def main():
    LR = 0.25
    EPOCHS = 300
    LAYERS = [784, 100, 10]

    dataPlotter = AccuracyPlotter(learn_rate=LR, epochs=EPOCHS, layers=LAYERS)
    neural_network = NeuralNetwork(layer_sizes=LAYERS, learn_rate=LR, epochs=EPOCHS, accuracy_plotter=dataPlotter)
    training_images, training_labels, testing_images, testing_labels = getData()
    neural_network.train(training_images, training_labels)
    print("Finished Training")
    neural_network.test(testing_images, testing_labels)
    print("Finished Testing")
    # dataPlotter.showPlot()
    seePerformance(neural_network, training_images, training_labels)


if __name__ == '__main__':
    main()
