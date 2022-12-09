import numpy as np
from random import shuffle
from sys import argv



def lossFunction(y, yprime):
    return (1/2)*(y-yprime) ^ 2


class NeuralNetwork:

    def __init__(self) -> None:
        self.weights = None
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def predict(self, input):
        for layer in self.layers:
            input = layer.forward_propagation(input)
        return 1 if input >= 0.5 else 0

    def lossPrime(self, y, yprime):
        return y-yprime

    def backProp(self, example, learning_rate):
        output = example[0]
        for layer in self.layers:
            output = layer.forward_propagation(output)
        # backward propagation
        error = self.lossPrime(output, example[1])
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate)

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
            self.weights -= learning_rate * weights_error
            self.bias -= learning_rate * output_error
            return input_error

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
        testData.append((np.array(listToAdd), (1 if float(terms[4]) != 0 else 0)))



# def backpropogation(weights, input, gamma):
#     #input is a pair with (data, label)

#     #forward propogation
#     weightUpdates = []
#     y = input[1]
#     input = input[0]
#     inputs = []
#     inputs.append(np.reshape(input, (-1,1)))
#     for weight in weights:
#         input = np.dot(input, weight)
#         if len(input) > 1:
#             inputs.append(np.reshape(input, (-1,1)))
#             input = sig(input)


#     error = lossFunctionPrime(y, input[0])


#     for it in reversed(range(len(weights))):
#         weighterror = np.dot(sig(inputs[it]), error)
#         error = np.multiply(np.dot(error, weights[it].T), sigPrime(inputs[it]).T)
#         weights[it] -= gamma* weighterror
#     return weightUpdates


if len(argv) > 1:
    numberOfNodes = int(argv[1])
    print(f"User input received, using {numberOfNodes} Neurons per layer.")
else:
    print("No user input, default is 5 Neurons per layer.")
    numberOfNodes = 5
def SGD(S, T = 100, gamma = 0.1):
    nn = NeuralNetwork()
    
    nn.addLayer(NeuralNetwork.Layer(4, numberOfNodes))
    nn.addLayer(NeuralNetwork.Layer(numberOfNodes, numberOfNodes))
    nn.addLayer(NeuralNetwork.Layer(numberOfNodes, 1))

    for i in range(T):
        gammat = gamma/(1+(gamma/0.085)*i)
        shuffle(S)
        for input in S:
            nn.backProp(input, gammat)

    return nn

NN = SGD(trainData)

print(f"Train Error {getErrorRate(NN, trainData)}")
print(f"Test Error {getErrorRate(NN, testData)}")

