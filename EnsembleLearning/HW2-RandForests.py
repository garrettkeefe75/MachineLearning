import sys
import math
import random
sys.path.insert(0, "..")
import DecisionTree.ID3 as ID3
import time
strt =time.time()


def RandomForest(S, attributes, label, depth=-1, variation="Entropy", weights=None, subsetSize=2):
    if weights == None:
        weights = [1/len(S)]*len(S)
    root = ID3.node()
    testLabel = S[0][len(S[0])-1]
    allLabelsSame = True
    for example in S:
        if example[len(example)-1] != testLabel:
            allLabelsSame = False
            break

    if allLabelsSame:
        root.setAttribute(testLabel)
        return root
    
    if depth == 0:
        root.setAttribute(ID3.mostCommonLabel(S, label))
        return root

    attributesSubset = dict(random.sample(list(attributes4.items()), subsetSize))
    bestInfoGain = 0
    A = None
    for key in attributesSubset.keys():
        index, values = attributesSubset.get(key)
        infoGain = ID3.IG(S, (index, values), label, variation, weights)
        #print(f"{key} : {infoGain}")
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            A = key
    
    if A == None:
        root.setAttribute(ID3.mostCommonLabel(S, label))
        return root

    attrIndex, values = attributes.get(A)
    root.setAttribute(A, attrIndex)
    for value in values:
        Sv = []
        SvWc = []
        i = 0

        for example in S:
            if example[attrIndex] == value:
                Sv.append(example)
                SvWc.append(weights[i])
            i += 1
        SvW = []
        for weight in SvWc:
            SvW.append(weight * len(weights)/len(Sv))

        if len(Sv) == 0:
            leaf = ID3.node()
            leaf.setAttribute(ID3.mostCommonLabel(S, label))
            root.append(value, leaf)
        else:
            attributeCopy = attributes.copy()
            del attributeCopy[A]
            root.append(value, RandomForest(Sv, attributeCopy, label, depth-1, variation, SvW))
    
    return root


def restoreUnknown(listToAdd, terms, index, attr):
    if terms[index] == 'unknown':
        listToAdd.append(attr)
    else:
        listToAdd.append(terms[index])

def useEnsemble(Ensemble, example):
    guess = 0
    for vote, root in Ensemble:
        guess += vote * root.getExampleGuess(example)
    guess = guess/len(Ensemble)
    return 1 if guess >= 0 else -1


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
if len(sys.argv) >= 2:
    T = int(sys.argv[1])
else:
    T = 5
Ensemble = []

for inc in range(T):
    bootstrapSamples = random.choices(exampleSet4, k=len(exampleSet4))
    root = RandomForest(bootstrapSamples, attributes4, label4, -1, "Entropy", weights, 2)
    err = 0
    i = 0
    for example in exampleSet4:
        if root.getExampleGuess(example) != example[len(example)-1]:
            err += weights[i]
        i += 1
    vote = 1/2 * math.log((1-err)/err)
    Ensemble.append((vote, root))

successes = 0
fails = 0
i = 0
for example in testData:
    if example[len(example)-1] == useEnsemble(Ensemble, example):
        successes += 1
    else:
        fails += 1
    i += 1

print(f"Error Rate: {fails/(successes+fails)}")

end = time.time()
print(f"Time to run {end - strt}")