import numpy as np
from copy import copy


def sigmoid(x, deriv=False):  # when derivative is true we pass values given to the sigmoid function
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def squaredError(difference,deriv=False):
        if deriv:
           return -difference
        return 0.5*(difference**2)

class Layer:        #helper class for MyNN
    #row numbers are index of input nodes,column numbers are index of output nodes
    def createWeights(self, rowNo, columnNo):
        return 2 * np.random.random((rowNo, columnNo)) - 1

    def __init__(self,inputSize, outputSize ):
        self.weights = self.createWeights(inputSize,outputSize)
        self.sigmoidedValues = np.zeros((1,outputSize))     #no need to lots of recomputation as output values used multiple times
        self.bias = np.random.random()

    def giveInput(self,input):
        normalValues = np.dot(input,self.weights)
        self.sigmoidedValues = sigmoid(np.array(normalValues))  #output values of this layer with given input
        return self.sigmoidedValues+self.bias

    def updateWeights(self,learningRate,postLayerEffects,preLayerSigmoidValue):
        """
        #one by one update
        inputSize,outputSize = self.weights.shape
        postLayerEffects = postLayerEffects[0]
        preLayerSigmoidValue = preLayerSigmoidValue[0]
        for inputNodeIndex in range(inputSize):
            for outputNodeIndex in range(outputSize):
                self.weights[inputNodeIndex][outputNodeIndex] = self.weights[inputNodeIndex][outputNodeIndex] - \
                                    learningRate*postLayerEffects[outputNodeIndex]*preLayerSigmoidValue[inputNodeIndex]
        """
        postLayerEffects = np.array(postLayerEffects)
        preLayerSigmoidValue = np.array(preLayerSigmoidValue).transpose()
        updateValues = learningRate*np.dot(preLayerSigmoidValue,postLayerEffects)
        self.weights = self.weights - updateValues


class MyNN:
    def __init__(self, learningRate, inputSize, outputSize, hiddenLayerSizes=None):
        self.learningRate = learningRate
        self.inputSize = inputSize      #use for validation
        layerSizes = []
        if hiddenLayerSizes is not None:
            layerSizes = copy(hiddenLayerSizes)
        layerSizes.append(outputSize)
        self.createLayers(layerSizes)

    def checkShape(self,askedArray,askedShape):     #for validation
        array = np.array[copy(askedArray)]
        if len(array.shape) == 1: array = np.array([array])
        return askedShape == array.shape

    def createLayers(self, layerSizes):
        #np.random.seed(1)   #reduces randomisation, just for test
        layers = []
        inputSize = self.inputSize
        for layerSize in layerSizes:
            layers.append(Layer(inputSize,layerSize))
            inputSize = layerSize
        self.layers = np.array(layers)

    def predict(self,input):
        if (not self.checkShape(input,(1,self.inputSize))):  return None
        prediction = None
        current = sigmoid(copy(input))
        for layer in self.layers:
            prediction = layer.giveInput(current)
            current = prediction
        return prediction

    def test(self,inputOutputPair,error=False):
        returnValues = []
        for input,output in inputOutputPair:
            returnValues.append(self.testOneSet(input,output,error))

    def testOneSet(self, input, output,error=False):    #to see error/difference of prediction over actual value
        if (not self.checkShape(input,(1,self.inputSize))) or \
            (not  self.checkShape(output,self.layers[-1].sigmoidedValues).shape):  return None
        prediction = self.predict(input)
        difference = output-prediction
        if error:
            return np.sum(squaredError(difference))
        return difference

    def train(self,inputOutputPair):
        returnValues = []
        for input,output in inputOutputPair:
            returnValues.append(self.trainOneSet(input,output))

    def trainOneSet(self,input,output):
        if (not self.checkShape(input,(1,self.inputSize))) or \
            (not  self.checkShape(output,self.layers[-1].sigmoidedValues).shape):  return None
        difference = self.testOneSet(input,output)
        layerEffects = self.setNodeEffectsToError(difference)
        self.backpropagation(layerEffects,input)
        return difference

    def getAllSigmoidValues(self):
        sigmoidValues = []
        for layer in self.layers:
            sigmoidValues.append(copy(layer.sigmoidedValues))
        return sigmoidValues

    def setNodeEffectsToError(self, difference):        #derivation of squared error by every node layer by layer
        #node effect over total error is sum of the multiplications of node's weights and weight's ending node's effect over total error
        allSigmoidValues = self.getAllSigmoidValues()
        layerEffects = [squaredError(difference, True)]     #last layer's nodes' effect
        for layerNo in range(len(self.layers) - 1, 0, -1):
            lastLayerEffects = layerEffects[0][0]
            currentLayerSigmoidDerivatedValues = sigmoid(allSigmoidValues[layerNo-1][0],True)
            currentLayerWeights = self.layers[layerNo].weights
            currentLayerEffects = []
            inputSize,outputSize = currentLayerWeights.shape
            for inputNodeNo in range(inputSize):
                nodeEffect = 0
                for outputNodeNo in range(outputSize):
                    nodeEffect += lastLayerEffects[outputNodeNo]*currentLayerWeights[inputNodeNo][outputNodeNo]*currentLayerSigmoidDerivatedValues[inputNodeNo]
                currentLayerEffects.append(nodeEffect)
            layerEffects.insert(0,[currentLayerEffects])
        return layerEffects



    def backpropagation(self, layerEffects,input):
        #weight effect over total error is multiplication of its derivation of starting node's sigmoided value and ending node's effect over total error
        #to update a weight, subtract the multiplication of its effect over total error and learning rate
        sigmoidValues = self.getAllSigmoidValues()
        for layerNo,layer in enumerate(self.layers):
            rate = self.learningRate
            layerEffect = layerEffects[layerNo]
            if layerNo ==0 :
                sigmoidValue = sigmoid(copy(input))
            else:
                sigmoidValue = sigmoidValues[layerNo-1]
            layer.updateWeights(rate,layerEffect,sigmoidValue)


