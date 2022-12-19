from NeuralNetwork import NeuralNetwork
import numpy as np
import pandas as pd
from sklearn import preprocessing as pp # I am a child
from sklearn.neural_network import MLPClassifier

from random import shuffle
import csv

le = pp.LabelEncoder()
columnValues = {}

df = pd.read_csv('../DecisionTree/income2022f/train_final.csv')

column = df["workclass"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("workclass", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

column = df["education"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("education", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

column = df["marital.status"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("marital.status", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

column = df["occupation"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("occupation", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

column = df["relationship"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("relationship", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

column = df["race"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("race", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

sexlabel = le.fit_transform(df['sex'])
df.drop("sex", axis=1, inplace=True)


column = df["native.country"]
columnValues["Holand-Netherlands"] = [0] * len(column)
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("native.country", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

df.drop("?", axis=1, inplace=True)

y_train = df["income>50K"].to_numpy()
df.drop("income>50K", axis=1, inplace=True)
df.sort_index(axis=1, inplace=True)
scaler = pp.MinMaxScaler(feature_range=(-1, 1))
scaler.fit(df)
names = df.columns
d = scaler.transform(df)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df["sex"] = sexlabel
x_train = scaled_df.to_numpy()
trainData = []
for i in range(len(x_train)):
    trainData.append((np.reshape(x_train[i],(1,-1)), y_train[i]))

df = pd.read_csv('../DecisionTree/income2022f/test_final.csv')

column = df["workclass"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("workclass", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

column = df["education"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("education", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

column = df["marital.status"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("marital.status", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

column = df["occupation"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("occupation", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

column = df["relationship"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("relationship", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

column = df["race"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("race", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

sexlabel = le.fit_transform(df['sex'])
df.drop("sex", axis=1, inplace=True)


column = df["native.country"]
for value in column.unique():
    columnValues[value] = [0] * len(column)
for i in range(len(column)):
    columnValues[column[i]][i] = 1


df.drop("native.country", axis=1, inplace=True)
for newColumn in columnValues.keys():
    df[newColumn] = columnValues[newColumn]
columnValues = {}

df.drop("ID", axis=1, inplace=True)
df.drop("?", axis=1, inplace=True)
df.sort_index(axis=1, inplace=True)


names = df.columns
d = scaler.transform(df)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df["sex"] = sexlabel
x_test = scaled_df.to_numpy()
testData = []
for i in range(len(x_test)):
    testData.append(np.reshape(x_test[i],(1,-1)))

# clf = MLPClassifier(random_state=1, max_iter=1000000, activation='tanh')
# clf.fit(x_train, y_train)
NN = NeuralNetwork(len(trainData[0][0][0]), 3, 69)

# for i in range(25):
#     NN.SGD(trainData[:1000], 50)
#     shuffle(trainData)
NN.SGD(trainData, 250)

answers = []
print(f"Train Error: {NN.getErrorRate(trainData)}")

i = 1
for test in testData:
    answers.append([i, NN.predictNonBinary(test)])
    i +=1

# arrayOfAnswers = clf.predict(x_test)
# for i in range(len(arrayOfAnswers)):
#     answers.append([i+1, arrayOfAnswers[i]])

with open('submission25-NeuralNetworks.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='\'', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['ID','Prediction'])
    for row in answers:
        writer.writerow(row)