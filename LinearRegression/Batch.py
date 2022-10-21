from numpy.linalg import norm
import numpy as np

def costFunc(weights, data):
    sum = 0
    for example in data:
        y = example[len(example)-1]
        subset = np.array(example[:7])
        thing = y - np.dot(np.transpose(weights), subset)
        sum += thing ** 2
    return sum/2

trainData = []

with open('./concrete/train.csv', 'r') as f:
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        for i in range(8):
            listToAdd.append(float(terms[i]))
        trainData.append(listToAdd)

testData = []

with open('./concrete/test.csv', 'r') as f:
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        for i in range(8):
            listToAdd.append(float(terms[i]))
        testData.append(listToAdd)

weights = np.zeros(7)
prevWeights = None
toleranceLevel = 10e-6
lr = 0.015

while True:
    prevWeights = weights
    sum = np.zeros(7)
    for example in trainData:
        y = example[len(example)-1]
        subset = np.array(example[:7])
        thing = y - np.dot(np.transpose(weights), subset)
        sum += thing * subset
    gradient = -sum
    weights = weights - lr*gradient
    print(costFunc(weights, trainData))

    #if True: break
    if norm((weights - prevWeights)) < toleranceLevel: break
print(weights)