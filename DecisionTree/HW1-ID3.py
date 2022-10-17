import math


class node:
    def __init__(self) -> None:
        self.nextNodes = {}
        self.attribute = None
        self.attributeIndex = -1

    def append(self, val, node):
        self.nextNodes[val] = node

    def setAttribute(self, attr, attrIndex=-1):
        self.attribute = attr
        self.attributeIndex = attrIndex

    def printNode(self, depth=0):
        thing = '|'*depth
        thing2 = '>'*(depth+1)
        print(thing + self.attribute)
        for key in self.nextNodes.keys():
            #print(thing2+key)
            self.nextNodes.get(key).printNode(depth+1)

    def testExample(self, example):
        #print(f"attribute: {self.attribute} branches: {self.nextNodes.keys()} target: {example[self.attributeIndex]}")
        if self.attributeIndex == -1:
            if example[len(example)-1] == self.attribute:
                return True
            else:
                return False
        return self.nextNodes.get(example[self.attributeIndex]).testExample(example)


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

def mostCommonLabel(S, label):
    mostCommonValue = None
    mostCommonValueOccurence = 0
    for value in label:
        i = 0
        for example in S:
            if example[len(example)-1] == value:
                i += 1
        if i > mostCommonValueOccurence:
            mostCommonValueOccurence = i
            mostCommonValue = value
    return mostCommonValue

exampleSet1 = []
attributes1 = {"x1": (0, ['0', '1']), "x2": (1, ['0', '1']),
"x3": (2, ['0', '1']), "x4": (3, ['0', '1'])}
label1 = ['0', '1']


with open('./trainBool.csv', 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        exampleSet1.append(terms)


exampleSet2 = []
attributes2 = {"O": (0, ['S', 'O', 'R']), "T": (1, ['H', 'M', 'C']),
"H": (2, ['H', 'N', 'L']), "W": (3, ['S', 'W'])}
label2 = ['+', '-']


with open('./trainTennis.csv', 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        exampleSet2.append(terms)


exampleSet3 = []
attributes3 = {"buying": (0, ['vhigh', 'high', 'med', 'low']),
"maint": (1, ['vhigh', 'high', 'med', 'low']),
"doors": (2, ['2', '3', '4', '5more']),
"persons": (3, ['2', '4', 'more']),
"lug_boot": (4, ['small','med','big']),
"safety": (5, ['low','med','high'])}
label3 = ['unacc', 'acc','good','vgood']


with open('./Car/train.csv', 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        exampleSet3.append(terms)

exampleSet4 = []
attributes4 = {"age": (0, ['1', '0']),
"job": (1, ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 
'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 
'technician', 'services']),
"marital": (2, ["married","divorced","single"]),
"education": (3, ["unknown","secondary","primary","tertiary"]),
"default": (4, ['yes','no']),
"balance": (5, ['1','0']),
"housing": (6, ["yes","no"]),
"loan": (7, ["yes","no"]),
"contact": (8, ["unknown","telephone","cellular"]),
"day": (9, ['1', '0']),
"month": (10, ["jan", "feb", "mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]),
"duration": (11, ['1','0']),
"campaign": (12, ['1','0']),
"pdays": (13, ['1','0']),
"previous": (14, ['1','0']),
"poutcome": (15, ["unknown","other","failure","success"])}
label4 = ['yes','no']

ages = []
balances = []
days = []
durations = []
campaigns = []
pdays = []
previous = []

with open('./bank/train.csv', 'r') as f:
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
        restoreUnknown(listToAdd, terms, 1, 'blue-collar')
        listToAdd.append(terms[2])
        restoreUnknown(listToAdd, terms, 3, 'secondary')
        listToAdd.append(terms[4])
        numericBoolean(listToAdd, terms, 5, balances)
        listToAdd.append(terms[6])
        listToAdd.append(terms[7])
        restoreUnknown(listToAdd, terms, 8, 'cellular')
        numericBoolean(listToAdd, terms, 9, days)
        listToAdd.append(terms[10])
        numericBoolean(listToAdd, terms, 11, durations)
        numericBoolean(listToAdd, terms, 12, campaigns)
        numericBoolean(listToAdd, terms, 13, pdays)
        numericBoolean(listToAdd, terms, 14, previous)
        restoreUnknown(listToAdd, terms, 15, 'failure')
        listToAdd.append(terms[16])
        exampleSet4.append(listToAdd)

def prob(val, S):
    i = 0
    for example in S:
        if example[len(example)-1] == val:
            i += 1
    if len(S)==0:
        return 0
    return i/len(S)


def Entropy(S, labelValues):
    sum = 0
    for value in labelValues:
        p = prob(value, S)
        if p != 0:
            sum += p*math.log(p, 2)
    return 0-sum


def ME(S, labelValues):
    mostCommonValueOccurence = 0
    if len(S) == 0:
        return 1
    for value in labelValues:
        i = 0
        for example in S:
            if example[len(example)-1] == value:
                i += 1
        if i > mostCommonValueOccurence:
            mostCommonValueOccurence = i
    return 1 - (mostCommonValueOccurence/len(S))
        
def GI(S, labelValues):
    sum = 0
    for value in labelValues:
        p = prob(value, S)
        sum += p*p
    return 1 - sum

def IG(S, A, labelValues, variation):
    attrIndex, values = A
    sum = 0
    for value in values:
        Sv = []

        for example in S:
            if example[attrIndex] == value:
                Sv.append(example)
        if variation == "ME":
            sum += (len(Sv)/len(S)) * ME(Sv, labelValues)
        elif variation == "GI":
            sum += (len(Sv)/len(S)) * GI(Sv, labelValues)
        else: 
            sum += (len(Sv)/len(S)) * Entropy(Sv, labelValues)
        

    if variation == "ME":
        return ME(S, labelValues) - sum
    elif variation == "GI":
        return GI(S, labelValues) - sum
    else:
        return Entropy(S, labelValues) - sum


def ID3(S, attributes, label, depth=-1, variation="Entropy"):  
    root = node()
    testLabel = S[0][len(S[0])-1]
    allLabelsSame = True
    for example in S:
        if example[len(example)-1] != testLabel:
            allLabelsSame = False
            break

    if allLabelsSame:
        root.setAttribute(testLabel)
        return root

    bestInfoGain = 0
    A = None
    for key in attributes.keys():
        index, values = attributes.get(key)
        infoGain = IG(S, (index, values), label, variation)
        #print(f"{key} : {infoGain}")
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            A = key
    
    if A == None or depth == 0:
        root.setAttribute(mostCommonLabel(S, label))
        return root

    attrIndex, values = attributes.get(A)
    root.setAttribute(A, attrIndex)
    for value in values:
        Sv = []

        for example in S:
            if example[attrIndex] == value:
                Sv.append(example)
        if len(Sv) == 0:
            leaf = node()
            leaf.setAttribute(mostCommonLabel(S, label))
            root.append(value, leaf)
        else:
            attributeCopy = attributes.copy()
            del attributeCopy[A]
            root.append(value, ID3(Sv, attributeCopy, label, depth-1, variation))

    return root

testData = []
with open('./bank/test.csv', 'r') as f:
    for line in f:
        listToAdd = []
        terms = line.strip().split(',')
        numericBoolean(listToAdd, terms, 0, ages)
        restoreUnknown(listToAdd, terms, 1, 'blue-collar')
        listToAdd.append(terms[2])
        restoreUnknown(listToAdd, terms, 3, 'secondary')
        listToAdd.append(terms[4])
        numericBoolean(listToAdd, terms, 5, balances)
        listToAdd.append(terms[6])
        listToAdd.append(terms[7])
        restoreUnknown(listToAdd, terms, 8, 'cellular')
        numericBoolean(listToAdd, terms, 9, days)
        listToAdd.append(terms[10])
        numericBoolean(listToAdd, terms, 11, durations)
        numericBoolean(listToAdd, terms, 12, campaigns)
        numericBoolean(listToAdd, terms, 13, pdays)
        numericBoolean(listToAdd, terms, 14, previous)
        restoreUnknown(listToAdd, terms, 15, 'failure')
        listToAdd.append(terms[16])
        testData.append(listToAdd)

for i in range(16):
    root = ID3(exampleSet4, attributes4, label4, i, "en")
    successes = 0
    fails = 0
    for example in testData:
        if root.testExample(example):
            successes += 1
        else:
            fails += 1
    print(f"Depth: {i+1} Error Rate: {fails/(successes+fails)}")
    
#root.printNode()