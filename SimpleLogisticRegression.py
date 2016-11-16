import numpy as np
from sklearn import datasets,linear_model

#loading data
iris = datasets.load_iris()

# filtering, only target = 0 and 1 remained
flt = np.logical_or(iris.target == 0 , iris.target == 1)
iris.target = iris.target[flt]
iris.data = iris.data[flt]

# sperate training and test
msk = np.random.rand(len(iris.target)) < 0.8
train_X = iris.data[msk]
train_Y = iris.target[msk]

test_X = iris.data[~msk]
test_Y = iris.target[~msk]

# creating model
model = linear_model.LogisticRegression(C = 1.0)
model.fit(train_X,train_Y)


# predict test
predVals = model.predict(test_X)

# checking acc
totalNum = len(test_Y)
truePredCount = sum(np.equal(predVals,test_Y))
print 'Acc: %f' % ( float(truePredCount) / float(totalNum))

# printing probabilities of instances in test
probs = model.predict_proba(test_X)
print 'probabilities of instances to belong a class'
print '-------------------------'
print '| class #1  | class #2  | pred  | true |'
print '----------------------------------------'
for i in xrange(len(test_Y)):
    p = probs[i]
    print '| %.6f  | %.6f  |   %d   |  %d   |' % (p[0],p[1],predVals[i],test_Y[i])