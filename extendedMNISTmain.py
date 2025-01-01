"""
Data used for this comes from https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset

@author: Edward Denton
"""
from os import listdir
import numpy as np
import pandas as pd
import random as rand
import cv2

from matplotlib import pyplot as plt

from NeuralNetwork import NeuralNetwork
from AccuracyPlotter import AccuracyPlotter

FILEPATH = "extMNIST/"


def resize_images():
    for filename in listdir(FILEPATH + "Img"):
        image = cv2.imread(FILEPATH + "Img/" + filename)
        resizedImage = cv2.resize(src=image, dsize=(28, 28))
        cv2.imwrite(FILEPATH + "newImg/re" + filename, resizedImage)


def processData():
    columnNames = [f"pixel{i}" for i in range(28 * 28)]
    columnNames.insert(0, "label")
    imgLabelDF = pd.read_csv(FILEPATH + "english.csv")
    dataRows = []

    for i in range(len(imgLabelDF)):
        image = cv2.imread(FILEPATH + "newImg/re" + imgLabelDF.iloc[i, 0].split("/")[1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = image.flatten()
        newRow = [imgLabelDF.iloc[i, 1]] + image.tolist()
        dataRows.append(newRow)

    dataDF = pd.DataFrame(dataRows, columns=columnNames)
    dataDF.to_csv(FILEPATH + "extMNISTdata.csv", index=False)


def getData():
    data = pd.read_csv(FILEPATH + "extMNISTdata.csv")
    data = np.array(data)

    training_data = []
    testing_data = []

    for i in range(0, len(data), 55):
        rows = data[i:i + 55]
        np.random.shuffle(rows)

        training_data.extend(rows[:45])
        testing_data.extend(rows[45:])

    training_data = np.array(training_data)
    testing_data = np.array(testing_data)

    testing_data = np.transpose(testing_data)
    testing_labels = testing_data[0]
    testing_images = np.array(testing_data[1:] / 255.0, dtype=np.float64)

    training_data = np.transpose(training_data)
    training_labels = training_data[0]
    training_images = np.array(training_data[1:] / 255.0, dtype=np.float64)

    return training_images, training_labels, testing_images, testing_labels


def main():
    LR = 0.05
    EPOCHS = 100
    LAYERS = [784, 128, 64, 62]
    BATCH_SIZE = 128

    # resize_images()
    # processData()
    dataPlotter = AccuracyPlotter(learn_rate=LR, epochs=EPOCHS, layers=LAYERS, batch_size=BATCH_SIZE)
    neural_network = NeuralNetwork(layer_sizes=LAYERS, learn_rate=LR, epochs=EPOCHS, accuracy_plotter=dataPlotter)
    training_images, training_labels, testing_images, testing_labels = getData()
    neural_network.train(training_images, training_labels, BATCH_SIZE)
    print("Finished Training")
    neural_network.test(testing_images, testing_labels)
    print("Finished Testing")
    dataPlotter.showPlot()
    # seePerformance(neural_network, training_images, training_labels)


if __name__ == '__main__':
    main()
