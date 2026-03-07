import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import random
import math
import os.path
from os import listdir
from os.path import isfile, join
from PIL import Image
import random
import time
np.seterr(divide = 'ignore') 

trainingData = []
trainingLabel = []
testData = []
testLabels = []

#checkpointFolder = r"\\DESKTOP-6BU9B33\Users\Manoj\Documents\Projects\MyProjects\MachineLearning\Keras\CnnImageColorize\model"
#checkpointFolder = r"D:\CnnImageColorize\model"
checkpointFolder = r"C:\Users\DELL\Documents\CnnImageColorizeGPU\model"
#predictionsFolder = r"D:\CnnImageColorize\images\predictions"
predictionsFolder = r"C:\Users\DELL\Documents\CnnImageColorizeGPU\images\predictions"
#os.chdir(r"C:\Users\Manoj\Documents\CnnImageColorizeGPU\images")
os.chdir(r"C:\Users\DELL\Documents\CnnImageColorizeGPU\images")
#os.chdir(r"D:\CnnImageColorize")
#os.chdir(r"\\DESKTOP-6BU9B33\Users\Manoj\Documents\Projects\MyProjects\MachineLearning\Keras\CnnImageColorize")

def getRecordCount(folderName):
    count = 0
    for dirpath, dirnames, filenames in os.walk(folderName):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            count += 1
    #print(folderName + " - " + str(count))
    return count

def getAllFileNames(folderName):
    count = 0
    fileNames = []
    for dirpath, dirnames, filenames in os.walk(folderName):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            fileNames.append(filename)
            count += 1
    #print(folderName + " - " + str(count))
    return filenames


totalRecordCount = getRecordCount("color_all")
trainRecordCount = getRecordCount("color_traindata")
testRecordCount = getRecordCount("color_testdata")

def getArrayForImage(fileName):
    data = []
    if(os.path.isfile(fileName) == False):
        print("FILE NUMBER Does not exist", fileName)
    image1 = Image.open(fileName).convert('RGB')
    pix1 = image1.load()
    width, height = image1.size
    row1 = []
    for rowIndex in range(width):
        col1 = []
        for colIndex in range(height):
            col1.append(pix1[rowIndex, colIndex])
        row1.append(col1)
    data.append(row1)
    return data[0]

def loadTrainingData(dataSize):
    global trainingData
    global trainingLabel
    trainingData = []
    trainingLabel = []
    randomIndices = []
    allIndices = []
    for index in range(trainRecordCount + 50):
        allIndices.append(index)
    randomIndices = random.choices(allIndices, k = dataSize)
    for index in randomIndices:
        dataFile = "grayscaled_traindata\\grayscaled_" + str(index) + ".jpg"
        if(os.path.isfile(dataFile) == False):
            #print("FILE NUMBER Does not exist", dataFile)
            continue
        labelFile = "color_traindata\\resized_" + str(index) + ".jpg"
        if(os.path.isfile(labelFile) == False):
            #print("FILE NUMBER Does not exist", labelFile)
            continue
        trainingData.append(getArrayForImage(dataFile))
        trainingLabel.append(getArrayForImage(labelFile))
    trainingData = np.array(trainingData)
    trainingLabel = np.array(trainingLabel)

def loadTestData(dataSize = -1):
    global testData
    global testLabels
    testData = []
    testLabels = []
    allFileNames = getAllFileNames("grayscaled_testdata")
    print("allFileNames ",  len(allFileNames))
    max = totalRecordCount
    allIndices = []
    for index in range(totalRecordCount):
        allIndices.append(index)
    random.shuffle(allIndices)
    for index in allIndices:
        if(("grayscaled_" + str(index) + ".jpg") in allFileNames):
            dataFile = "grayscaled_testdata\\grayscaled_" + str(index) + ".jpg"
            labelFile = "color_testdata\\resized_" + str(index) + ".jpg"
            testData.append(getArrayForImage(dataFile))
            testLabels.append(getArrayForImage(labelFile))
            if(dataSize > 0):
                if(len(testData) > dataSize):
                    break

    testData = np.array(testData)
    testLabels = np.array(testLabels)

def printPredictions(predictionSize):
    print("\nPredictions...iteration ", str(iteration))
    global testData
    global testLabels

    for index in range(len(testData)):
        try:
            tempIndex = random.randint(0, len(testData))
            prediction = model.predict(testData[tempIndex:tempIndex + 1])
            test_loss, test_acc = model.evaluate(testData[tempIndex:tempIndex + 1],  testLabels[tempIndex:tempIndex + 1], verbose = 1)

            #print('prediction.shape ', prediction[0].shape)   
            random_array = prediction[0].astype(np.uint8)  #For dense unit 3
            imgP = Image.fromarray(random_array)#.convert('RGB')
            imgP.save(predictionsFolder + "\\TEST_" + str(iteration) + "_" + str(tempIndex) + "_colorized_" + str(int(test_loss)) + ".jpg")

            temp = testLabels[tempIndex:tempIndex + 1]
            temp = temp[0].astype(np.uint8)  #For dense unit 3    
            #print(temp.shape)
            imgP = Image.fromarray(temp)
            imgP.save(predictionsFolder + "\\TEST_" + str(iteration) + "_" + str(tempIndex) + "_original_color_" + str(int(test_loss)) + ".jpg")

            #print(test_acc)

            if(index > predictionSize):
                break
        except Exception as e: 
            print(e)

    for index in range(len(trainingData)):
        try:
            tempIndex = random.randint(0, len(trainingData))
            prediction = model.predict(trainingData[tempIndex:tempIndex + 1])
            test_loss, test_acc = model.evaluate(trainingData[tempIndex:tempIndex + 1],  trainingLabel[tempIndex:tempIndex + 1], verbose = 1)

            #print('prediction.shape ', prediction[0].shape)   
            random_array = prediction[0].astype(np.uint8)  #For dense unit 3
            imgP = Image.fromarray(random_array)#.convert('RGB')
            imgP.save(predictionsFolder + "\\TRAIN_" + str(iteration) + "_" + str(tempIndex) + "_colorized_" + str(int(test_loss)) + ".jpg")

            temp = trainingLabel[tempIndex:tempIndex + 1]
            temp = temp[0].astype(np.uint8)  #For dense unit 3    
            #print(temp.shape)
            imgP = Image.fromarray(temp)
            imgP.save(predictionsFolder + "\\TRAIN_" + str(iteration) + "_" + str(tempIndex) + "_original_color_" + str(int(test_loss)) + ".jpg")

            #print(test_acc)

            if(index > predictionSize):
                break
        except Exception as e: 
            print(e)



boardSize = 100
trainingDataSize = 100
iterationCount = 20000
trainingEpochs = 15

print("Loading test data...")
loadTestData(trainingDataSize)

print('testData.shape ', testData.shape)               
print('testLabels.shape ', testLabels.shape)   

model = models.Sequential()

model.add(layers.Conv2D(boardSize, (5, 5), activation='relu', padding="same", input_shape=(boardSize, boardSize, 3)))
#model.add(layers.Conv2D(boardSize, (25, 25), activation='relu', padding="same", input_shape=(boardSize, boardSize, 3)))
#model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(boardSize * 2, (3, 3), activation='relu', padding="same"))
#model.add(layers.Conv2D(boardSize * 2, (10, 10), activation='relu', padding="same"))
#model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(boardSize * 4, (3, 3), activation='relu', padding="same"))
#model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))

#model.add(layers.Conv2D(boardSize, (3, 3), activation='relu', padding="same"))
#model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.25))

#model.add(layers.Flatten())
#model.add(layers.Dense(units=3, activation='relu'))
#model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.5))

model.add(layers.Dense(units=3, activation='relu'))
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE), metrics=[tf.keras.metrics.CosineSimilarity()])
                #metrics=['mean_squared_logarithmic_error'])

iteration = 0
while iteration < iterationCount:
    print("Loading training data...")
    loadTrainingData(trainingDataSize)
    print('trainData.shape ', trainingData.shape)               
    print('trainLabels.shape ', trainingLabel.shape)               

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointFolder + "\\cp.ckpt", save_weights_only=True, save_freq='epoch', verbose=2)

    model.summary()

    checkpoints = os.listdir(checkpointFolder)
    print("Checkpoints - ", checkpoints)
    if(len(checkpoints) > 0):
        print("FOUND CHECKPOINT LOADING CHECKPOINT LOADING CHECKPOINT")
        model.load_weights(checkpointFolder + "\\cp.ckpt")
    else:
        print("MODEL NOT FOUND MODEL NOT FOUND MODEL NOT FOUND")
        #exit()
    history = model.fit(trainingData, trainingLabel, epochs = trainingEpochs, 
                        validation_data=(testData, testLabels), callbacks=[cp_callback], verbose = 1)

    #test_loss, test_acc = model.evaluate(testData,  testLabels, verbose=2)
    #print(test_acc)

    #model.save(checkpointFolder)

    printPredictions(15)
    time.sleep(300)
    iteration += 1

exit()

