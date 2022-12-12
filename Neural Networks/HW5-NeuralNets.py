import numpy as np
from random import shuffle
from sys import argv


class NeuralNetwork:

    class Layer:
        def sig(self, x):
            return 1/(1+np.exp(-x))
        def sigPrime(self, x):
            return 1/(1+np.exp(-x))*(1-1/(1+np.exp(-x)))

        def __init__(self, input_size, output_size) -> None:
            self.weights = np.random.rand(input_size, output_size) - 0.5
            self.bias = np.random.rand(1, output_size) - 0.5
            self.numNodes = output_size

        def forward_propagation(self, input_data):
            self.input1 = input_data
            self.input2 = np.dot(self.input1, self.weights) + self.bias
            self.output = self.sig(self.input2)
            return self.output

        # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
        def backward_propagation(self, output_error, learning_rate):
            output_error = self.sigPrime(self.input2) * output_error
            input_error = np.dot(output_error, self.weights.T)
            weights_error = np.dot(self.input1.T, output_error)
            # dBias = output_error

            # update parameters
            #print(weights_error)
            self.weights -= learning_rate * weights_error
            self.bias -= learning_rate * output_error
            return input_error
    
    class FinalLayer(Layer):
        def forward_propagation(self, input_data):
            self.input = input_data
            self.output = np.dot(self.input, self.weights) + self.bias
            return self.output

        # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
        def backward_propagation(self, output_error, learning_rate):
            input_error = np.dot(output_error, self.weights.T)
            weights_error = np.dot(self.input.T, output_error)
            # dBias = output_error

            # update parameters
            #print(weights_error)
            self.weights -= learning_rate * weights_error
            self.bias -= learning_rate * output_error
            return input_error
    
    def __init__(self, numLayers = 0, sizeOfInput = 0, numberOfNodes = 0) -> None:
        self.weights = None
        self.layers = []
        if numLayers != 0:
            self.layers.append(NeuralNetwork.Layer(sizeOfInput, numberOfNodes))
        for i in range(numLayers-2):
            self.layers.append(NeuralNetwork.Layer(numberOfNodes, numberOfNodes))
        if numLayers != 0:
            self.layers.append(NeuralNetwork.FinalLayer(numberOfNodes, 1))

    def addLayer(self, layer):
        self.layers.append(layer)

    def predict(self, input):
        for layer in self.layers:
            input = layer.forward_propagation(input)
        return 1 if input >= 0.5 else 0
    
    def lossFunction(self, y, yprime):
        return (1/2)*(y-yprime)**2

    def lossPrime(self, y, yprime):
        return y-yprime

    def backProp(self, example, learning_rate):
        output = example[0]
        for layer in self.layers:
            output = layer.forward_propagation(output)
        # backward propagation
        #print(self.lossFunction(output, example[1]))
        error = self.lossPrime(output, example[1])
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate)

def getErrorRate(NN, testData):
    successes = 0
    fails = 0
    for test, y in testData:
        if NN.predict(test) == y:
            successes += 1
        else:
            fails += 1
    return fails/(successes+fails)

trainData = []

with open('../Perceptron/bank-note/bank-note/train.csv', 'r') as f:
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        for i in range(4):
            listToAdd.append(float(terms[i]))
        trainData.append((np.array([listToAdd]), (1 if float(terms[4]) != 0 else 0)))

testData = []

with open('../Perceptron/bank-note/bank-note/test.csv', 'r') as f:
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        for i in range(4):
            listToAdd.append(float(terms[i]))
        testData.append((np.array([listToAdd]), (1 if float(terms[4]) != 0 else 0)))

if len(argv) > 1:
    numberOfNodes = int(argv[1])
    print(f"User input received, using {numberOfNodes} Neurons per layer.")
else:
    print("No user input, default is 5 Neurons per layer.")
    numberOfNodes = 5

def SGD(S, numberOfNodes, T, gamma = 0.1, d = 0.085):
    nn = NeuralNetwork(3,4,numberOfNodes)

    for i in range(T):
        gammat = gamma/(1+(gamma/d)*i)
        shuffle(S)
        for input in S:
            #found example that did weight updates during back propagation, decided that made more sense.
            nn.backProp(input, gammat)

    return nn

if len(argv) > 2:
    T = int(argv[2])
    print(f"Using {T} epochs.")
else:
    T = 100
    print("Using 100 epochs.")

# NN = NeuralNetwork(3, 4, 5)
# NN.backProp(trainData[0], 0.01)


NN = SGD(trainData, numberOfNodes, T)

print(f"Train Error {getErrorRate(NN, trainData)}")
print(f"Test Error {getErrorRate(NN, testData)}")

