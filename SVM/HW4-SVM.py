import numpy
from sys import path
from random import shuffle
import scipy.optimize

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
        listToAdd.append(1)
        trainData.append((numpy.array(listToAdd), (1 if float(terms[4]) != 0 else -1)))

testData = []

with open('../Perceptron/bank-note/bank-note/test.csv', 'r') as f:
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        for i in range(4):
            listToAdd.append(float(terms[i]))
        listToAdd.append(1)
        testData.append((numpy.array(listToAdd), (1 if float(terms[4]) != 0 else -1)))

Cs = [100/873, 500/873, 700/873]

def PrimalSVM(y0, a, C, T = 100, trainData = trainData):
    weights = numpy.array([0]*5)
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
                w0 = weights.copy()
                w0[4] = 0
            else:
                w0 = (1-lr)*w0
    return weights




def DualSVM(C, trainData = trainData):
    def dualObjectiveFunc(alpha, X, y):
        X = numpy.matrix(X)
        yMatrix = y*numpy.ones(shape=(len(y), len(y)))
        alphaMatrix = alpha * numpy.ones(shape=(len(alpha), len(alpha)))
        Inner = (yMatrix*yMatrix.transpose()) * (alphaMatrix*alphaMatrix.transpose()) * (X*X.transpose())
        dualObjFunc = 0.5* Inner.sum() - sum(alpha)
        return dualObjFunc
    def constraint(alpha, y, C):
        sum = 0
        for i in range(len(alpha)):
            
            if 0 <= alpha[i] <= C:
                sum += alpha[i]*y[i]
            else:
                return 1
        return sum

    y = []
    X = []
    for Xi,yi in trainData:
        y.append(yi)
        X.append(Xi[:4])
    y = numpy.array(y)
    weights = numpy.array([0]*4)
    weights = numpy.asarray(weights, dtype = numpy.float64)
    constraints = [{'type': 'eq', 'fun': constraint, 'args': (y, C)}]
    Solution = scipy.optimize.minimize(dualObjectiveFunc, x0=numpy.zeros(shape=(len(X),)), args=(X,y), method="SLSQP", constraints=constraints)
    print(Solution["success"])
    for i in range(len(X)):
        weights = weights + (Solution["x"][i]*y[i]*X[i])
    bias = 0
    weights = numpy.asarray(weights).reshape(-1)
    for i in range(len(X)):
        
        bias += y[i] - weights.dot(X[i])
    bias = bias/len(X)
    weights = numpy.append(weights, [bias])
    return weights


print("Primal SVM ---- y0/(1+(y0*t/a)) schedule")
for C in Cs:
    w = PrimalSVM(0.01,0.0085, C)
    print(f"C: {C}  Weights: {w}   Train Error Rate: {getErrorRate(w, trainData)}   Test Error rate: {getErrorRate(w, testData)}")
print("\nPrimal SVM ---- y0/(1+t) schedule")
for C in Cs:
    w = PrimalSVM(0.01, 0, C)
    print(f"C: {C}  Weights: {w}   Train Error Rate: {getErrorRate(w, trainData)}   Test Error rate: {getErrorRate(w, testData)}")

print("\n\nDual SVM:")
for C in Cs:
    w = DualSVM(C)
    print(f"C: {C}  Weights: {w}   Train Error Rate: {getErrorRate(w, trainData)}   Test Error rate: {getErrorRate(w, testData)}")