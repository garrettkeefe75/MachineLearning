import numpy as np
from sys import argv
from NeuralNetwork import NeuralNetwork


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

if len(argv) > 2:
    T = int(argv[2])
    print(f"Using {T} epochs.")
else:
    T = 100
    print("Using 100 epochs.")

# NN = NeuralNetwork(3, 4, 5)
# NN.backProp(trainData[0], 0.01)
NN = NeuralNetwork(len(trainData[0][0][0]), 2, numberOfNodes)
NN.SGD(trainData, T)

print(f"Train Error {NN.getErrorRate(trainData)}")
print(f"Test Error {NN.getErrorRate(testData)}")

