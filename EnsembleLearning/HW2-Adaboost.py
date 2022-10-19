import sys
import math
import numpy
sys.path.insert(0, "..")
import DecisionTree.ID3 as ID3
import time
strt =time.time()


def restoreUnknown(listToAdd, terms, index, attr):
    if terms[index] == 'unknown':
        listToAdd.append(attr)
    else:
        listToAdd.append(terms[index])


def numericBoolean(listToAdd, terms, index, sortedSet):
    if terms[index] > sortedSet[int(len(sortedSet)/2)]:
        listToAdd.append('1')
    else:
        listToAdd.append('0')


exampleSet4 = []
attributes4 = {"age": (0, ['1', '0']),
               "job": (1, ['admin.', 'unknown', 'unemployed', 'management', 'housemaid',
                           'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired',
                           'technician', 'services']),
               "marital": (2, ["married", "divorced", "single"]),
               "education": (3, ["unknown", "secondary", "primary", "tertiary"]),
               "default": (4, ['yes', 'no']),
               "balance": (5, ['1', '0']),
               "housing": (6, ["yes", "no"]),
               "loan": (7, ["yes", "no"]),
               "contact": (8, ["unknown", "telephone", "cellular"]),
               "day": (9, ['1', '0']),
               "month": (10, ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
               "duration": (11, ['1', '0']),
               "campaign": (12, ['1', '0']),
               "pdays": (13, ['1', '0']),
               "previous": (14, ['1', '0']),
               "poutcome": (15, ["unknown", "other", "failure", "success"])}
label4 = [1, -1]

ages = []
balances = []
days = []
durations = []
campaigns = []
pdays = []
previous = []

with open('../DecisionTree/bank/train.csv', 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        ages.append(terms[0])
        balances.append(terms[5])
        days.append(terms[9])
        durations.append(terms[11])
        campaigns.append(terms[12])
        pdays.append(terms[13])
        previous.append(terms[14])

    ages.sort()
    balances.sort()
    days.sort()
    durations.sort()
    campaigns.sort()
    pdays.sort()
    previous.sort()
    f.seek(0)
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        numericBoolean(listToAdd, terms, 0, ages)
        # restoreUnknown(listToAdd, terms, 1, 'blue-collar')
        listToAdd.append(terms[1])
        listToAdd.append(terms[2])
        # restoreUnknown(listToAdd, terms, 3, 'secondary')
        listToAdd.append(terms[3])
        listToAdd.append(terms[4])
        numericBoolean(listToAdd, terms, 5, balances)
        listToAdd.append(terms[6])
        listToAdd.append(terms[7])
        # restoreUnknown(listToAdd, terms, 8, 'cellular')
        listToAdd.append(terms[8])
        numericBoolean(listToAdd, terms, 9, days)
        listToAdd.append(terms[10])
        numericBoolean(listToAdd, terms, 11, durations)
        numericBoolean(listToAdd, terms, 12, campaigns)
        numericBoolean(listToAdd, terms, 13, pdays)
        numericBoolean(listToAdd, terms, 14, previous)
        # restoreUnknown(listToAdd, terms, 15, 'failure')
        listToAdd.append(terms[15])
        listToAdd.append(1 if terms[16] == 'yes' else -1)
        exampleSet4.append(listToAdd)


testData = []
with open('../DecisionTree/bank/test.csv', 'r') as f:
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        numericBoolean(listToAdd, terms, 0, ages)
        # restoreUnknown(listToAdd, terms, 1, 'blue-collar')
        listToAdd.append(terms[1])
        listToAdd.append(terms[2])
        # restoreUnknown(listToAdd, terms, 3, 'secondary')
        listToAdd.append(terms[3])
        listToAdd.append(terms[4])
        numericBoolean(listToAdd, terms, 5, balances)
        listToAdd.append(terms[6])
        listToAdd.append(terms[7])
        # restoreUnknown(listToAdd, terms, 8, 'cellular')
        listToAdd.append(terms[8])
        numericBoolean(listToAdd, terms, 9, days)
        listToAdd.append(terms[10])
        numericBoolean(listToAdd, terms, 11, durations)
        numericBoolean(listToAdd, terms, 12, campaigns)
        numericBoolean(listToAdd, terms, 13, pdays)
        numericBoolean(listToAdd, terms, 14, previous)
        # restoreUnknown(listToAdd, terms, 15, 'failure')
        listToAdd.append(terms[15])
        listToAdd.append(1 if terms[16] == 'yes' else -1)
        testData.append(listToAdd)

weights = [1/len(exampleSet4)]*len(exampleSet4)
T = 5
H = [0] * len(exampleSet4)
Ensemble = []

for inc in range(T):
    err = 0
    root = ID3.ID3(exampleSet4, attributes4, label4, 1, "Entropy", weights)
    i = 0
    for example in exampleSet4:
        if root.getExampleGuess(example) != example[len(example)-1]:
            err += weights[i]
        i += 1
    vote = 1/2 * math.log((1-err)/err)
    # root.printNode()
    #print(vote)
    weightCopy = []
    Hcopy = []
    i = 0
    for example in exampleSet4:
        h = root.getExampleGuess(example)
        weightCopy.append(
            weights[i] * math.exp(-vote * example[len(example)-1]*h))
        Hcopy.append(H[i] + vote*h)
        # if h == example[len(example) - 1]:
        #     print(math.exp(-vote * example[len(example)-1]*h))
        i += 1
    s = sum(weightCopy)
    H = Hcopy
    weights = [float(w)/s for w in weightCopy]
    Ensemble.append((vote, root))
H = numpy.sign(H)

successes = 0
fails = 0
i = 0
for example in testData:
    guess = 0
    for vote, root in Ensemble:
        guess += vote * root.getExampleGuess(example)
    if guess >= 0:
        guess = 1 
    else:
        guess = -1
    if example[len(example)-1] == guess:
        successes += 1
    else:
        fails += 1
    i += 1

print(f"Error Rate: {fails/(successes+fails)}")

end = time.time()
print(f"Time to run {end - strt}")
