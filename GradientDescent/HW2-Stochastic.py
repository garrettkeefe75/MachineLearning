import random
from numpy.linalg import norm
import numpy as np
import csv

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
lr = 0.01
it = 0
figure1 = []
figure1.append(['steps', 'cost func'])

while True:
    it += 1
    example = random.choice(trainData)
    y = example[len(example)-1]
    subset = np.array(example[:7])
    thing = y - np.dot(np.transpose(weights), subset)
    prevWeights = weights
    weights = weights + (lr * thing * subset)
    figure1.append([it, costFunc(weights, trainData)])
    if norm((weights - prevWeights)) < toleranceLevel: break
print(f"weights: {weights}  learning rate: {lr}  Test cost func: {costFunc(weights, testData)}")
with open('figure1_Stochastic.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='\'', quoting=csv.QUOTE_MINIMAL)
    for row in figure1:
        writer.writerow(row)