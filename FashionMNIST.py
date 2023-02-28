import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import timeit
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import ast
import matplotlib.pyplot as plt


def main():
    tf.random.set_seed(50)
    np.random.seed(50) 
    hParams = {
             'datasetProportion': 1.0,
             'numEpochs': 20,
             'denseLayers': [512, 10],
             'valProportion': 0.1,
             'experimentName': "512_10"}   
    #oneHiddenLayer(get10ClassData(hParams),hParams)
    dataSubsets = get10ClassData(hParams)
    #denseNN(dataSubsets, hParams)
    trainResults, testResults = denseNN(get10ClassData(hParams),hParams)
    writeExperimentalResults(hParams, trainResults, testResults)
    readExperimentalResults("512_10")

    



def get10ClassData(hParams):
    proportion = hParams['datasetProportion']
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 
    x_train, y_train = correspondingShuffle(x_train, y_train)
    x_test, y_test = correspondingShuffle(x_test, y_test)
    
    if proportion != 1.0:
        num_samples = int(x_train.shape[0] * proportion)
        x_train = x_train[:num_samples]
        y_train = y_train[:num_samples]
        num_samples = int(x_test.shape[0] * proportion)
        x_test = x_test[:num_samples]
        y_test = y_test[:num_samples]
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    num_samples = x_train.shape[0]
    num_inputs = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(num_samples, num_inputs)
    x_test = x_test.reshape(x_test.shape[0], num_inputs)
    
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)    
    
    x_val = x_train[:int(hParams['valProportion'] * x_train.shape[0]),:]
    x_train = x_train[int(hParams['valProportion'] * x_train.shape[0]):,:]
    y_val = y_train[:int(hParams['valProportion'] * y_train.shape[0])]
    y_train = y_train[int(hParams['valProportion'] * y_train.shape[0]):]
    if hParams['valProportion'] != 0.0:
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test    
        

def oneHiddenLayer(dataSubsets, hParams):
    x_train, y_train, x_test, y_test = dataSubsets
    
    startTime = timeit.default_timer()
    layer = Sequential()
    layer.add(Dense(128, activation='relu'))
    layer.add(Dense(10))
    layer.compile(loss = SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])    
    layer.fit(x = x_train,y = y_train, epochs = hParams['numEpochs'])
    training_time = timeit.default_timer() - startTime
    
    print(layer.summary())
    print(layer.count_params())
    
    startTime = timeit.default_timer()
    evalResults = layer.evaluate(x_test, y_test)
    testing_time = timeit.default_timer() - startTime
    
    print("Training time:", training_time)
    print("Testing time:", testing_time)
    print("Testing set accuracy:", evalResults)
    
    
def denseNN(dataSubsets, hParams):
    if hParams['valProportion'] != 0.0:
        x_train, y_train, x_val, y_val, x_test, y_test = dataSubsets
    else:
        x_train, y_train, x_test, y_test = dataSubsets  
    denseLayers = hParams['denseLayers']
    
    startTime = timeit.default_timer()
   
    model = Sequential()
    for units in denseLayers[:-1]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(denseLayers[-1]))
    model.compile(loss = SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])    
    hist = model.fit(x = x_train,y = y_train,validation_data=(x_val, y_val) if
                                    hParams['valProportion']!=0.0
                                    else None, epochs = hParams['numEpochs'], verbose = 1)
    
    training_time = timeit.default_timer() - startTime
    
    print(model.summary())
    print(model.count_params())
    print(hist.history)
    hParams['paramCount'] = model.count_params()
    
    startTime = timeit.default_timer()
    
    evalResults = model.evaluate(x_test, y_test, verbose = 1)
    
    testing_time = timeit.default_timer() - startTime
    
    #print("Training time:", training_time)
    #print("Testing time:", testing_time)
    #print("Testing set accuracy:", evalResults)
    return hist.history, evalResults

    
def correspondingShuffle(x,y):
    indices = tf.range(start = 0, limit = tf.shape(x)[0])
    shuffled_indices = tf.random.shuffle(indices)
    
    shuffled_x = tf.gather(x, shuffled_indices)
    shuffled_y = tf.gather(y, shuffled_indices)
    return shuffled_x, shuffled_y

def writeExperimentalResults(hParams, trainResults, testResults):
    f = open("results/" + hParams['experimentName'] + ".txt", "w")
    f.write(str(hParams)+"\n")
    f.write(str(trainResults)+"\n")
    f.write(str(testResults))
    f.close()
    
def readExperimentalResults(nameOfFile):
    f = open("results/" + nameOfFile + ".txt","r") 
    results = f.read().split("\n")
    hParams = ast.literal_eval(results[0])
    trainResults = ast.literal_eval(results[1])
    testResults = ast.literal_eval(results[2])
    return hParams, trainResults, testResults

def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
    fig, ax = plt.subplots()
    y = np.array(yList).transpose()
    ax.plot(x, y)
    ax.set(xlabel=xLabel, title=title)
    plt.legend(yLabelList, loc='best', shadow=True)
    ax.grid()
    yLabelStr = "__" + "__".join([label for label in yLabelList])
    filepath = "results/" + title + " " + yLabelStr + ".png"
    fig.savefig(filepath)
    print("Figure saved in", filepath)    
    
def processResults():
    hParams, trainResults, testResults = readExperimentalResults("512_10")
    itemsToPlot = ['accuracy', 'val_accuracy']
    plotCurves(x=np.arange(0, hParams['numEpochs']),yList=[trainResults[item] for item in itemsToPlot],xLabel="Epoch",yLabelList=itemsToPlot,title=hParams['experimentName'])
    itemsToPlot = ['loss', 'val_loss']
    plotCurves(x=np.arange(0, hParams['numEpochs']),yList=[trainResults[item] for item in itemsToPlot],xLabel="Epoch",yLabelList=itemsToPlot,title=hParams['experimentName'])    

def buildValAccuracyPlot():
    experiments = ['128_10','4_10','32_10','512_10','256_128_64_64_10']
    fig, ax = plt.subplots()
    for i in experiments:
        hParams, trainResults, testResults = readExperimentalResults(i)
        x = np.arange(0,hParams['numEpochs'])
        y = np.array(trainResults['val_accuracy']).transpose()
        ax.plot(x,y,label=i)
        ax.set(xlabel = "Epoch", ylabel = "Validation Accuracy", title = "Validation Accuracy Plot")
    plt.legend(loc='best', shadow=True)
    filepath = "results/" + "val_accuracy" + ".png"
    fig.savefig(filepath)

def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
    plt.figure()
    plt.scatter(xList,yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if pointLabels != []:
        for i, label in enumerate(pointLabels):
            plt.annotate(label, (xList[i], yList[i]))
    filepath = "results/" + filename + ".png"
    plt.savefig(filepath)
    print("Figure saved in", filepath)
  
def buildTestAccuracyPlot():
    experiments = ['128_10','4_10','32_10','512_10','256_128_64_64_10']
    param_counts = []
    test_accuracies = []
    for i in experiments:
        hParams, trainResults, testResults = readExperimentalResults(i)
        param_counts.append(hParams['paramCount'])
        test_accuracies.append(testResults[1])
    plotPoints(param_counts, test_accuracies, pointLabels=experiments, xLabel="parameter count", yLabel="Test Set Accuracy", title="Test Set Accuracy")


#main()
#processResults()
buildValAccuracyPlot()
buildTestAccuracyPlot()