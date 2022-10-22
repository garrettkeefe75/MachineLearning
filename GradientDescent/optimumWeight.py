import numpy as np
from numpy.linalg import inv

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

X = []
Y = []
for example in trainData:
    X.append(example[:7])
    Y.append(example[7])
X = np.array(X)
Y = np.array(Y)
inter = X.transpose().dot(X)
inter2 = X.transpose().dot(Y)
interinv = inv(inter)
final = interinv.dot(inter2)
print(costFunc(final, trainData))