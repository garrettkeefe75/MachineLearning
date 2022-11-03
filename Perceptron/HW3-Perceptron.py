import numpy
from sys import argv
from random import shuffle

def prediction(weights, test):
    return 1 if weights.transpose().dot(test) >= 0 else -1

def votedPrediction(weightVoteList, test):
    sum = 0
    for weight, c in weightVoteList:
        sum += c * (1 if weight.transpose().dot(test) >= 0 else -1)
    return 1 if sum >= 0 else -1

trainData = []

with open('./bank-note/bank-note/train.csv', 'r') as f:
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        for i in range(4):
            listToAdd.append(float(terms[i]))
        trainData.append((numpy.array(listToAdd), (1 if float(terms[4]) != 0 else -1)))

testData = []

with open('./bank-note/bank-note/test.csv', 'r') as f:
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        for i in range(4):
            listToAdd.append(float(terms[i]))
        testData.append((numpy.array(listToAdd), (1 if float(terms[4]) != 0 else -1)))


def StandardPerceptron(T, r):
    weights = numpy.array([0]*4)
    for i in range(T):
        shuffle(trainData)
        for x, y in trainData:
            if (y * (weights.transpose().dot(x))) <= 0:
                weights = weights + (r*y*x)
    return weights

def VotedPerceptron(T, r):
    weights = numpy.array([0]*4)
    weightVoteList = []
    C = 0
    for i in range(T):
        for x, y in trainData:
            if (y * (weights.transpose().dot(x))) <= 0:
                weightVoteList.append((weights, C))
                weights = weights + (r*y*x)
                C = 1
            else:
                C += 1
    weightVoteList.append((weights, C))
    return weightVoteList

def AveragedPerceptron(T, r):
    weights = numpy.array([0]*4)
    a = numpy.array([0]*4)
    for i in range(T):
        for x, y in trainData:
            if (y * (weights.transpose().dot(x))) <= 0:
                weights = weights + (r*y*x)
            a = a + weights
    return a

T = 10
r = 0.1
if len(argv) > 3:
    T = int(argv[2])
    r = float(argv[3])
elif len(argv) > 2:
    T = int(argv[2])

useAltPred = False
if len(argv) > 1 and argv[1] == "Standard":
    weights = StandardPerceptron(T, r)
elif len(argv) > 1 and argv[1] == "Voted":
    weights = VotedPerceptron(T, r)
    useAltPred = True
elif len(argv) > 1 and argv[1] == "Averaged":
    weights = AveragedPerceptron(T, r)
else:
    print("Please specify which Perceptron algorithm you want to use.")
    exit()

successes = 0
fails = 0

for x, y in testData:        
    if not useAltPred and prediction(weights, x) == y:
        successes += 1
    elif useAltPred and votedPrediction(weights, x) == y:
        successes += 1
    else:
        fails += 1

if useAltPred:
    o = numpy.array([0]*4)
    for weight, vote in weights:
        print(f"Weight: {weight}  Count: {vote}")
        #o = o + vote*weight
    #print(o)
    print(f"Prediction Error: {fails/(successes+fails)}")
else:
    print(f"Weights: {weights}  Prediction Error: {fails/(successes+fails)}")