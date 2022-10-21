import sys
import math
sys.path.insert(0, "..")
import DecisionTree.ID3 as ID3
import time
import csv
strt =time.time()


def getError(data, Ensemble):
    successes = 0
    fails = 0
    for example in data:
        if example[len(example)-1] == useEnsemble(Ensemble, example):
            successes += 1
        else:
            fails += 1
    return fails/(successes+fails)

def getErrorFullEnsemble(data, Ensemble):
    helper = [(0,0)]*len(Ensemble)
    allErrorForData = []
    for example in data:
        i = 0
        for guess in useFullEnsemble(Ensemble, example):
            successes, fails = helper[i]
            if guess == example[len(example)-1]:
                successes += 1
            else:
                fails += 1
            helper[i] = (successes, fails)
            i += 1
    for successes, fails in helper:
        allErrorForData.append(fails/(fails+successes))
    return allErrorForData

def restoreUnknown(listToAdd, terms, index, attr):
    if terms[index] == 'unknown':
        listToAdd.append(attr)
    else:
        listToAdd.append(terms[index])

def useEnsemble(Ensemble, example):
    guess = 0
    for vote, root in Ensemble:
        guess += vote * root.getExampleGuess(example)
    if guess >= 0: guess = 1 
    else: guess = -1
    return guess

def useFullEnsemble(Ensemble, example):
    guess = 0
    i = 0
    guesses = []
    for vote, root in Ensemble:
        guess += vote * root.getExampleGuess(example)
        i+=1
        interGuess = guess/i
        guesses.append(1 if interGuess >= 0 else -1)
    return guesses



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
figure1 = []
figure2 = []
figure1.append(['T', 'train error', 'test error'])
figure2.append(['T', 'train error', 'test error'])


for inc in range(T):
    root = ID3.ID3(exampleSet4, attributes4, label4, 1, "Entropy", weights)
    err = 0
    i = 0
    for example in exampleSet4:
        if root.getExampleGuess(example) != example[len(example)-1]:
            err += weights[i]
        i += 1
    vote = 1/2 * math.log((1-err)/err)
    #root.printNode()
    #print(vote)
    weightCopy = []
    i = 0
    for example in exampleSet4:
        h = root.getExampleGuess(example)
        weightCopy.append(
            weights[i] * math.exp(-vote * example[len(example)-1]*h))
        i += 1
    s = sum(weightCopy)
    weights = [float(w)/s for w in weightCopy]
    Ensemble.append((vote, root))
    figure2.append([inc+1, getError(exampleSet4, [(vote, root)]), getError(testData, [(vote, root)])])

trainerrors = getErrorFullEnsemble(exampleSet4, Ensemble)
testerrors = getErrorFullEnsemble(testData, Ensemble)
for it in range(T):
    figure1.append([it+1, trainerrors[it], testerrors[it]])


with open('figure1_Adaboost.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='\'', quoting=csv.QUOTE_MINIMAL)
    for row in figure1:
        writer.writerow(row)

with open('figure2_Adaboost.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='\'', quoting=csv.QUOTE_MINIMAL)
    for row in figure2:
        writer.writerow(row)


end = time.time()
#root = ID3.ID3(exampleSet4, attributes4, label4, 2, "Entropy")
print(f"Time to run {end - strt}")
