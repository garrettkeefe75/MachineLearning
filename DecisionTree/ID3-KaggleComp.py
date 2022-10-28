import sys
import math
import random
sys.path.insert(0, "..")
import ID3
from collections import Counter
import csv

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
        root.setAttribute(ID3.mostCommonLabel(S, label, weights))
        return root

    if len(attributes) >= subsetSize:
        attributesSubset = dict(random.sample(list(attributes.items()), subsetSize))
    else:
        attributesSubset = attributes
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
        root.setAttribute(ID3.mostCommonLabel(S, label, weights))
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
            leaf.setAttribute(ID3.mostCommonLabel(S, label, weights))
            root.append(value, leaf)
        else:
            attributeCopy = attributes.copy()
            del attributeCopy[A]
            root.append(value, RandomForest(Sv, attributeCopy, label, depth-1, variation, SvW, subsetSize))
    
    return root

def useEnsemble(Ensemble, example):
    guess = 0
    for vote, root in Ensemble:
        guess += vote * root.getExampleGuess(example)
    guess = guess/len(Ensemble)
    return guess
 
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def restoreUnknown(listToAdd, terms, index, attr):
    if terms[index] == '?':
        listToAdd.append(attr)
    else:
        listToAdd.append(terms[index])

def numericBoolean(term, sortedSet):
    if int(term) > sortedSet[int(len(sortedSet)/2)]:
        return 1
    else:
        return 0


exampleSet = []
attributes = {"age": (0,[0,1]),
"workclass": (1,['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 
                'State-gov', 'Without-pay', 'Never-worked']),
"fnlgwt": (2,[0,1]),
"education":(3,['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 
                'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', 
                '5th-6th', 'Preschool']),
"education.num": (4,[0,1]),
"marital.status": (5,['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 
                    'Married-spouse-absent', 'Married-AF-spouse']),
"occupation": (6,['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 
                'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 
                'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']),
"relationship": (7,['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']),
"race": (8,['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']),
"sex": (9,['Female', 'Male']),
"capital-gain": (10,[0,1]),
"capital-loss": (11,[0,1]),
"hours-per-week": (12,[0,1]),
"native-country": (13,['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 
                    'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 
                    'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 
                    'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 
                    'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 
                    'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])}
label = [0,1]

ages = []
workClasses = []
fnlwgts = []
educationNums = []
occupations = []
capitalGains = []
capitalLosses = []
hoursPerWeek = []
nativeCountries = []



with open('./income2022f/train_final.csv', 'r') as f:
    f.readline()
    for line in f:
        terms = line.strip().split(',')
        ages.append(int(terms[0]))
        workClasses.append(terms[1])
        fnlwgts.append(int(terms[2]))
        educationNums.append(int(terms[4]))
        occupations.append(terms[6])
        capitalGains.append(int(terms[10]))
        capitalLosses.append(int(terms[11]))
        hoursPerWeek.append(int(terms[12]))
        nativeCountries.append(terms[13])
    
    ages.sort()
    fnlwgts.sort()
    educationNums.sort()
    capitalGains.sort()
    capitalLosses.sort()
    hoursPerWeek.sort()
    mostCommonWorkClass = most_frequent(workClasses)
    mostCommonOccupation = most_frequent(occupations)
    mostCommonCountry = most_frequent(nativeCountries)
    #unknowns on workclass, occupation, native-country
    f.seek(0)
    f.readline()
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        listToAdd.append(numericBoolean(terms[0],ages))
        restoreUnknown(listToAdd, terms, 1, mostCommonWorkClass)
        listToAdd.append(numericBoolean(terms[2], fnlwgts))
        listToAdd.append(terms[3])
        listToAdd.append(numericBoolean(terms[4], educationNums))
        listToAdd.append(terms[5])
        restoreUnknown(listToAdd, terms, 6, mostCommonOccupation)
        listToAdd.append(terms[7])
        listToAdd.append(terms[8])
        listToAdd.append(terms[9])
        listToAdd.append(numericBoolean(terms[10], capitalGains))
        listToAdd.append(numericBoolean(terms[11], capitalLosses))
        listToAdd.append(numericBoolean(terms[12], hoursPerWeek))
        restoreUnknown(listToAdd, terms, 13, mostCommonCountry)
        listToAdd.append(int(terms[14]))
        exampleSet.append(listToAdd)


testData = []
with open('./income2022f/test_final.csv', 'r') as f:
    f.readline()
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        listToAdd.append(numericBoolean(terms[1],ages))
        restoreUnknown(listToAdd, terms, 2, mostCommonWorkClass)
        listToAdd.append(numericBoolean(terms[3], fnlwgts))
        listToAdd.append(terms[4])
        listToAdd.append(numericBoolean(terms[5], educationNums))
        listToAdd.append(terms[6])
        restoreUnknown(listToAdd, terms, 7, mostCommonOccupation)
        listToAdd.append(terms[8])
        listToAdd.append(terms[9])
        listToAdd.append(terms[10])
        listToAdd.append(numericBoolean(terms[11], capitalGains))
        listToAdd.append(numericBoolean(terms[12], capitalLosses))
        listToAdd.append(numericBoolean(terms[13], hoursPerWeek))
        restoreUnknown(listToAdd, terms, 14, mostCommonCountry)
        testData.append(listToAdd)


Ensemble = []
weights = [1/len(exampleSet)]*len(exampleSet)
k = 4
for inc in range(500):
    bootstrapSamples = random.choices(exampleSet, k=len(exampleSet))
    root = RandomForest(bootstrapSamples, attributes, label, -1, "Entropy", weights, k)
    err = 0
    i = 0
    for example in exampleSet:
        if root.getExampleGuess(example) != example[len(example)-1]:
            err += weights[i]
        i += 1
    vote = 1/2 * math.log((1-err)/err)
    Ensemble.append((vote, root))

# for i in range(13):
#     root = ID3.ID3(exampleSet, attributes, label, i, "Entropy")
#     successes = 0
#     fails = 0
#     for example in exampleSet:
#         if root.testExample(example):
#             successes += 1
#         else:
#             fails += 1
#     print(f"Depth: {i+1} Error Rate: {fails/(successes+fails)}")

# root = ID3.ID3(exampleSet, attributes, label, 5)
answers = []
i = 1
for example in testData:
    answers.append([i, useEnsemble(Ensemble,example)])
    i += 1
# successes = 0
# fails = 0
# for example in exampleSet:
#     if root.testExample(example):
#         successes += 1
#     else:
#         fails += 1
# print(f"Depth: 2 Error Rate: {fails/(successes+fails)}")

with open('submission12-RandomForests.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='\'', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['ID','Prediction'])
    for row in answers:
        writer.writerow(row)
