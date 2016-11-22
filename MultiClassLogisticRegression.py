'''
Author: mahat
'''

from sklearn import linear_model, datasets, metrics
import numpy as np

# loading data
iris = datasets.load_iris()

# sperate training and test
msk = np.random.rand(len(iris.target)) < 0.8
train_X = iris.data[msk]
train_Y = iris.target[msk]

test_X = iris.data[~msk]
test_Y = iris.target[~msk]

# regularization
reg = 0.01

# multi-class stretegies
# One Vs All(Rest) (OVA) -- default
OVRModel = linear_model.LogisticRegression(C=reg, multi_class='ovr', solver='newton-cg',class_weight=None)
OVRModel.fit(train_X, train_Y)

# predict test
predVals = OVRModel.predict(test_X)

# checking acc
totalNum = len(test_Y)
truePredCount = sum(np.equal(predVals, test_Y))
print 'Report for One Vs Rest Strategy'
print metrics.classification_report(predVals, test_Y)

# One Vs One
# One Vs OnO is a voting strategy based on comparing each model to another model


# Creating voter models which compares one class to another one
setOfClasses = set(iris.target)
numberOfClasses = len(setOfClasses)
classList = list(setOfClasses)
modelList = []
for i in classList:
    for j in classList:
        if i < j:
            flt = np.logical_or(np.equal(train_Y, i), np.equal(train_Y, j))
            subX = train_X[flt]
            subY = train_Y[flt]
            subModel = linear_model.LogisticRegression(C=reg)
            subModel.fit(subX, subY)
            modelList.append(subModel)

# Counting of votes of models for determining classes of test set

predVals = []
for testIns in test_X:
    votes = np.zeros(numberOfClasses)
    # print votes

    for currModel in modelList:
        predVal = currModel.predict([testIns])
        tmpIn = classList.index(predVal)
        votes[tmpIn] = votes[tmpIn] + 1

    predVals.append(classList[votes.argmax()])


# print results
totalNum = len(test_Y)
truePredCount = 0
for pred, trueClass in zip(predVals,test_Y):
    if pred == trueClass:
        truePredCount = truePredCount + 1

print 'Report for One Vs One Strategy'
print metrics.classification_report(predVals, test_Y)
