import math


class node:
    def __init__(self) -> None:
        self.nextNodes = {}
        self.attribute = None

    def append(self, val, node):
        self.nextNodes[val] = node

    def setAttribute(self, attr):
        self.attribute = attr

    def printNode(self, depth):
        thing = '|'*depth
        thing2 = '>'*(depth+1)
        print(thing + self.attribute)
        for key in self.nextNodes.keys():
            #print(thing2+key)
            self.nextNodes.get(key).printNode(depth+1)


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
            sum += p*math.log(p)
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

def IG(S, A, labelValues):
    attrIndex, values = A
    sum = 0
    for value in values:
        Sv = []

        for example in S:
            if example[attrIndex] == value:
                Sv.append(example)
        sum += (len(Sv)/len(S)) * Entropy(Sv, labelValues)

    return Entropy(S, labelValues) - sum


def ID3(S, attributes, label, depth=-1):        
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
        infoGain = IG(S, (index, values), label)
        print(f"{key} : {infoGain}")
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            A = key
    
    if A == None or depth == 0:
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
        root.setAttribute(mostCommonValue)
        return root

    root.setAttribute(A)
    attrIndex, values = attributes.get(A)
    for value in values:
        Sv = []

        for example in S:
            if example[attrIndex] == value:
                Sv.append(example)
        if len(Sv) == 0:
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
            leaf = node()
            leaf.setAttribute(mostCommonValue)
            root.append(value, leaf)
        else:
            attributeCopy = attributes.copy()
            del attributeCopy[A]
            root.append(value, ID3(Sv, attributeCopy, label, depth-1))

    return root


root = ID3(exampleSet2, attributes2, label2,0)
#print(Entropy(exampleSet2,label2))



#root.printNode(0)