# Might make your life easier for appending to lists
from collections import defaultdict

# Third party libraries
import numpy as np
import sklearn.metrics
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
# Only needed if you plot your confusion matrix
import matplotlib.pyplot as plt

# our libraries
from lib.partition import split_by_day
import lib.file_utilities as util


# Any other modules you create


def dolphin_classifier(data_directory):
    """
    Neural net classification of dolphin echolocation clicks to species
    :param data_directory:  root directory of data
    :return:  None
    """
    # get files to a dict sorted by date
    fileInput = util.get_files(data_directory, ext=".czcc")
    recordingInfoTypes = util.parse_files(fileInput)
    sortedRecordingInfoTypes = split_by_day(recordingInfoTypes)

    # split into training and test days
    sorted = list(sortedRecordingInfoTypes)
    train, test = train_test_split(sorted)

    #-----BEGIN DATA PROCESSING-----#

    #------TRAINING DATA------------#
    xTrain = [] #Feature Array
    yTrain = [] #Label Array
    #Populate feature and label arrays.
    k = 0
    for i in train:
        for j in range(len(sortedRecordingInfoTypes[i])):
            for k in range(len(sortedRecordingInfoTypes[i][j].features)):
                xTrain.append(sortedRecordingInfoTypes[i][j].features[k])
                yTrain.append(sortedRecordingInfoTypes[i][j].label)

    #Changing the arrays into numpy arrays for processing
    yTrain = [[i] for i in yTrain]
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

    #Creating one hot labels for our data. [GG,LO] example: [1,0] means it's a GG
    oneHotY = np.zeros((yTrain.shape[0],2))
    for i in range(len(yTrain)):
        if yTrain[i] == "Gg":
            oneHotY[i][0] = 1
        elif yTrain[i] == "Lo":
            oneHotY[i][1] = 1
        else:
            print("Error: Invalid Label")

    # ------TEST DATA------------#
    xTest = []  # Feature Array
    yTest = []  # Label Array
    # Populate feature and label arrays.
    k = 0
    for i in test:
        for j in range(len(sortedRecordingInfoTypes[i])):
            for k in range(len(sortedRecordingInfoTypes[i][j].features)):
                xTest.append(sortedRecordingInfoTypes[i][j].features[k])
                yTest.append(sortedRecordingInfoTypes[i][j].label)

    # Changing the arrays into numpy arrays for processing
    yTest = [[i] for i in yTest]
    xTest = np.array(xTest)
    yTest = np.array(yTest)

    # Creating one hot labels for our data. [GG,LO] example: [1,0] means it's a GG
    weightDict = {0:0,1:0}
    oneHotYtest = np.zeros((yTest.shape[0], 2))
    for i in range(len(yTest)):
        if yTest[i] == "Gg":
            oneHotYtest[i][0] = 1
            weightDict[0] = weightDict[0] + 1
        elif yTest[i] == "Lo":
            oneHotYtest[i][1] = 1
            weightDict[1] = weightDict[1] + 1
        else:
            print("Error: Invalid Label test")

    #Making the layer
    # making the model
    model = Sequential(name="Rocky")

    model.add(InputLayer(input_shape=(20,)))
    model.add(Dense(units=100, activation="relu",kernel_regularizer = regularizers.l2(.01)))
    model.add(Dense(units=100, activation="relu",kernel_regularizer = regularizers.l2(.01)))
    model.add(Dense(units=100, activation="relu",kernel_regularizer = regularizers.l2(.01)))
    model.add(Dense(2, activation="softmax"))
    #Compile the model
    model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    model.summary()
    #Training the model
    model.fit(xTrain,oneHotY,epochs = 4,class_weight = weightDict)

    #Evalutating the model
    results = model.evaluate(xTest,oneHotYtest)
    predictions = model.predict(xTest)
    predictions = np.argmax(predictions,axis=1)
    predictions2 = []
    for i in range(len(predictions)):
        if predictions[i] == 0:
            predictions2.append("Gg")
        elif predictions[i] == 1:
            predictions2.append("Lo")
        else:
            print("Error in predictions")


    plt.ion()  # enable interactive plotting

    #creating the confusion matrix and displaying
    confMatrix = sklearn.metrics.confusion_matrix(yTest,predictions2,labels = ["Gg","Lo"])
    disp = sklearn.metrics.ConfusionMatrixDisplay(confMatrix,display_labels = ["Gg","Lo"])
    disp.plot()
    plt.show()

    #Calcluating and printing out the error rate:

    errorRate = 1-results[1]
    print(f'Error Rate: {errorRate}')


    use_onlyN = np.Inf  # debug, only read this many files for each species

    # if NotImplementedError:
    #     implement()


if __name__ == "__main__":
    data_directory = "/home/kylekrueger/Downloads/Project/dolphin_features"  # root directory of data
    dolphin_classifier(data_directory)

 #---------CODE GRAVEYARD-------------#
# joined = np.hstack((xTrain, yTrain))
# print(xTrain.shape)
# print(oneHotY.shape)
# print(f'x train: {xTrain}')
# print(f'y train: {oneHotY}')
# print(f'hstack: {joined}')

# tf_xTrain = tf.convert_to_tensor(xTrain)
# tf_yTrain = tf.convert_to_tensor(yTrain)
# print(tf_xTrain)
# print(tf_yTrain)
# lengthForShape = len(tf_xTrain)

# print(f'length: {len(sortedRecordingInfoTypes[i][j].features)} \n\n\n')# print(len(sortedRecordingInfoTypes[train[1]][0].features))
# print()
# print()
# print()

# The training feature vectors should be N x 20, where N is the number of echolocation feature vectors.
# The labels should be Nx2 and contain one-hot information.
# The gradient descent algorithm uses the labels to compute the loss.

# making tensors
# for i in train:
#     for j in range(len(sortedRecordingInfoTypes[i])):
#         if sortedRecordingInfoTypes[i][j].label == "Lo":
#             print("shawty got lo")
#         if sortedRecordingInfoTypes[i][j].label == "Gg":
#             print("good game")

# print(f'train:      {train} \n')
    # print()
    # print(f'test:      {test} \n')
    # print(sortedRecordingInfoTypes)
    # print()
    # print()
    # print()

    #oneColdYtest = np.zeros((oneHotYtest.shape[0]))
    # for i in range(len(oneHotYtest)):
    #     if oneHotYtest[i][1] == 1:
    #         oneColdYtest[i] = 1

# oneHotpredictions = np.zeros((predictions.shape[0], 2))
# for i in range(len(predictions)):
#     if predictions[i] == 0:
#         oneHotpredictions[i][0] = 1
#     elif yTrain[i] == 1:
#         oneHotpredictions[i][1] = 1
#     else:
#         print("Error: Invalid Label")
# print(oneHotpredictions)

#Creating the layer
    # layer = Dense(
    #     units=100,
    #     kernel_regularizer=regularizers.L2(0.001),
    #     activation = "relu"
    # )
    # inputLayer = InputLayer(
    #     input_shape=(20,),
    #     kernel_regularizer=regularizers.L2(0.001)
    # )
    # outputLayer = Dense(
    #     units = 2,
    #     activation = "softmax"
    # )
