import numpy
from sys import path
from random import shuffle
import scipy

def prediction(weights, test):
    return 1 if weights.transpose().dot(test) >= 0 else -1

def getErrorRate(weights, testData):
    successes = 0
    fails = 0
    for test, y in testData:
        if prediction(weights, test) == y:
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
        trainData.append((numpy.array(listToAdd), (1 if float(terms[4]) != 0 else -1)))

testData = []

with open('../Perceptron/bank-note/bank-note/test.csv', 'r') as f:
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        for i in range(4):
            listToAdd.append(float(terms[i]))
        testData.append((numpy.array(listToAdd), (1 if float(terms[4]) != 0 else -1)))

Cs = [100/873, 500/873, 700/873]

def PrimalSVM(y0, a, C, T = 100, trainData = trainData):
    weights = numpy.array([0]*4)
    w0 = weights.copy()
    N = len(trainData)
    for t in range(T):
        if a == 0:
            lr = y0/(1+t)
        else:
            lr = y0/(1+(y0*t/a))
        shuffle(trainData)
        for x, y in trainData:
            if y * weights.transpose().dot(x) <= 1:
                weights = weights-lr*(w0 - (C*N*y*x))
                w0 = (weights/numpy.linalg.norm(weights))
            else:
                w0 = (1-lr)*w0
    return weights

def DualSVM(y0, a, C, T = 100, trainData = trainData):
    weights = numpy.array([0]*4)
    w0 = weights.copy()
    N = len(trainData)
    for t in range(T):
        if a == 0:
            lr = y0/(1+t)
        else:
            lr = y0/(1+(y0*t/a))
        shuffle(trainData)
        for x, y in trainData:
            if y * weights.transpose().dot(x) <= 1:
                weights = weights-lr*(w0 - (C*N*y*x))
                w0 = (weights/numpy.linalg.norm(weights))
            else:
                w0 = (1-lr)*w0
    return weights

for C in Cs:
    w = PrimalSVM(0.01,0.0085, C)
    print(f"Weights: {w}   Train Error Rate: {getErrorRate(w, trainData)}   Test Error rate: {getErrorRate(w, testData)}")
for C in Cs:
    w = PrimalSVM(0.01, 0, C)
    print(f"Weights: {w}   Train Error Rate: {getErrorRate(w, trainData)}   Test Error rate: {getErrorRate(w, testData)}")