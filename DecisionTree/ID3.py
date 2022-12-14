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
        print(thing + str(self.attribute))
        for key in self.nextNodes.keys():
            self.nextNodes.get(key).printNode(depth+1)

    def testExample(self, example):
        #print(f"attribute: {self.attribute} branches: {self.nextNodes.keys()} target: {example[self.attributeIndex]}")
        n = self
        while n.attributeIndex != -1:
            n = n.nextNodes.get(example[n.attributeIndex])
        if example[len(example)-1] == n.attribute:
            return True
        else:
            return False

    def getExampleGuess(self, example):
        n = self
        while n.attributeIndex != -1:
            n = n.nextNodes.get(example[n.attributeIndex])
        return n.attribute

def mostCommonLabel(S, label, weights):
    mostCommonValue = None
    mostCommonValueOccurence = 0
    for value in label:
        sum = 0
        i = 0
        for example in S:
            if example[len(example)-1] == value:
                sum += weights[i]
            i += 1
        if sum > mostCommonValueOccurence:
            mostCommonValueOccurence = sum
            mostCommonValue = value
    return mostCommonValue

def prob(val, S, weights):
    retValue = 0
    i = 0
    for example in S:
        if example[len(example)-1] == val:
            retValue += weights[i]
        i += 1
    return retValue


def Entropy(S, labelValues, weights):
    sum = 0
    for value in labelValues:
        p = prob(value, S, weights)
        if p != 0:
            sum += p*math.log(p, 2)
    return 0-sum


def ME(S, labelValues, weights):
    mostCommonValueOccurence = 0
    if len(S) == 0:
        return 1
    for value in labelValues:
        sum = 0
        i = 0
        for example in S:
            if example[len(example)-1] == value:
                sum += weights[i]
            i += 1
        if sum > mostCommonValueOccurence:
            mostCommonValueOccurence = sum
    return 1 - mostCommonValueOccurence
        
def GI(S, labelValues, weights):
    sum = 0
    for value in labelValues:
        p = prob(value, S, weights)
        sum += p*p
    return 1 - sum

def IG(S, A, labelValues, variation, weights):
    attrIndex, values = A
    sum = 0
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
        if variation == "ME":
            sum += (len(Sv)/len(S)) * ME(Sv, labelValues, SvW)
        elif variation == "GI":
            sum += (len(Sv)/len(S)) * GI(Sv, labelValues, SvW)
        else: 
            sum += (len(Sv)/len(S)) * Entropy(Sv, labelValues, SvW)
        

    if variation == "ME":
        return ME(S, labelValues, weights) - sum
    elif variation == "GI":
        return GI(S, labelValues, weights) - sum
    else:
        return Entropy(S, labelValues, weights) - sum


def ID3(S, attributes, label, depth=-1, variation="Entropy", weights=None):
    if weights == None:
        weights = [1/len(S)]*len(S)
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
    
    if depth == 0:
        root.setAttribute(mostCommonLabel(S, label, weights))
        return root

    bestInfoGain = 0
    A = None
    for key in attributes.keys():
        index, values = attributes.get(key)
        infoGain = IG(S, (index, values), label, variation, weights)
        #print(f"{key} : {infoGain}")
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            A = key
    
    if A == None:
        root.setAttribute(mostCommonLabel(S, label, weights))
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
            leaf = node()
            leaf.setAttribute(mostCommonLabel(S, label, weights))
            root.append(value, leaf)
        else:
            attributeCopy = attributes.copy()
            del attributeCopy[A]
            root.append(value, ID3(Sv, attributeCopy, label, depth-1, variation, SvW))

    return root