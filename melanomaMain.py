"""
Data used for this comes from https://www.kaggle.com/competitions/siim-isic-melanoma-classification/



@author: Edward Denton
"""

from os import listdir
import numpy as np
import pandas as pd
import cv2
import concurrent.futures


from NeuralNetwork import NeuralNetwork
from AccuracyPlotter import AccuracyPlotter

FILEPATH = "melanomaData/"


def resize_and_save_image(image_path, save_path):
    image = cv2.imread(image_path)
    resizedImage = cv2.resize(src=image, dsize=(128, 128))
    cv2.imwrite(save_path, resizedImage)


def resize_images():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for filename in listdir(FILEPATH + "test"):
            image_path = FILEPATH + "test/" + filename
            save_path = FILEPATH + "reTest/re" + filename
            futures.append(executor.submit(resize_and_save_image, image_path, save_path))

        for filename in listdir(FILEPATH + "train"):
            image_path = FILEPATH + "train/" + filename
            save_path = FILEPATH + "reTrain/re" + filename
            futures.append(executor.submit(resize_and_save_image, image_path, save_path))

        for future in concurrent.futures.as_completed(futures):
            pass


def processData():
    columnNames = []
    redPixels = [f"pixel{i} (R)" for i in range(128 * 128)]
    greenPixels = [f"pixel{i} (G)" for i in range(128 * 128)]
    bluePixels = [f"pixel{i} (B)" for i in range(128 * 128)]
    for i in range(128 * 128):
        columnNames.append(redPixels[i])
        columnNames.append(greenPixels[i])
        columnNames.append(bluePixels[i])
    columnNames.insert(0, "label")

    dfTrainCSV = pd.read_csv(FILEPATH + "train.csv")
    dataRows = []
    for i in range(len(dfTrainCSV)):
        image = cv2.imread(FILEPATH + "reTrain/re" + dfTrainCSV.iloc[i, 0] + ".jpg")
        image = image.flatten()
        label = 0 if dfTrainCSV.iloc[i, 6] == "benign" else 1
        newRow = [label] + image.tolist()
        dataRows.append(newRow)

    dataDF = pd.DataFrame(dataRows, columns=columnNames)
    print(dataDF)
    dataDF.to_csv(FILEPATH + "completeMelanomaData.csv", index=False)
    print("Done")


def shrinkData():
    data = pd.read_csv(FILEPATH + "completeMelanomaData.csv")
    data.sort_values(by="label", ascending=False, inplace=True)

    malignant_rows = data[data["label"] == 1].to_numpy()
    benign_rows = data[data["label"] == 0].to_numpy()

    print(malignant_rows.shape)
    print(benign_rows.shape)
    np.random.shuffle(malignant_rows)
    np.random.shuffle(benign_rows)
    benign_rows = benign_rows[:3000]

    combined_rows = np.vstack((malignant_rows, benign_rows))
    combined_df = pd.DataFrame(combined_rows, columns=data.columns)
    combined_df.to_csv(FILEPATH + "melanomaData.csv", index=False)


def getData():
    malignant_train = 400
    benign_train = 400

    data = pd.read_csv(FILEPATH + "melanomaData.csv")
    data.sort_values(by="label", ascending=False, inplace=True)

    malignant_rows = data[data["label"] == 1].to_numpy()
    benign_rows = data[data["label"] == 0].to_numpy()

    np.random.shuffle(malignant_rows)
    np.random.shuffle(benign_rows)

    training_data = np.vstack((malignant_rows[:malignant_train], benign_rows[:benign_train]))
    testing_data = np.vstack((malignant_rows[malignant_train:], benign_rows[benign_train:]))

    np.random.shuffle(training_data)
    np.random.shuffle(testing_data)

    testing_data = np.transpose(testing_data)
    testing_labels = testing_data[0]
    testing_images = np.array(testing_data[1:] / 255.0, dtype=np.float64)

    training_data = np.transpose(training_data)
    training_labels = training_data[0]
    training_images = np.array(training_data[1:] / 255.0, dtype=np.float64)

    return training_images, training_labels, testing_images, testing_labels


def main():
    # TODO: Redo the Layers code so I can specify between Conv2D, Pool2D, and Dense layers and their dimensions
    LR = 0.0005
    EPOCHS = 100
    LAYERS = [49152, 1024, 512, 2]
    BATCH_SIZE = 512

    # shrinkData()
    # resize_images()
    # processData()
    dataPlotter = AccuracyPlotter(learn_rate=LR, epochs=EPOCHS, layers=LAYERS, batch_size=BATCH_SIZE)
    neural_network = NeuralNetwork(layer_sizes=LAYERS, learn_rate=LR, epochs=EPOCHS, accuracy_plotter=dataPlotter)
    training_images, training_labels, testing_images, testing_labels = getData()
    print("Data has been gathered")
    neural_network.train(training_images, training_labels, BATCH_SIZE)
    print("Finished Training")
    neural_network.test(testing_images, testing_labels)
    print("Finished Testing")
    dataPlotter.showPlot()
    # seePerformance(neural_network, training_images, training_labels)


if __name__ == '__main__':
    main()
