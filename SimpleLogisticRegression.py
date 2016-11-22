'''
Author: mahat
'''

import numpy as np
import pandas as pd
from sklearn import datasets,linear_model, metrics

#loading data
from sklearn.model_selection import train_test_split

import Utils

[X, Y,colNames] = Utils.getTitanicDataSet()

# sperate training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# creating model
model = linear_model.LogisticRegression(C = 1.0, penalty='l2')
model.fit(X_train,y_train)


# predict test
predicted = model.predict(X_test)

# checking acc
print 'Acc: %f' % metrics.accuracy_score(y_test, predicted)

# printing probabilities of instances in test
probs = model.predict_proba(X_test)
print 'probabilities of instances to belong a class'
print '-------------------------'
print '| class #1  | class #2  | pred  | true |'
print '----------------------------------------'
for i in xrange(len(y_test)):
    p = probs[i]
    print '| %.6f  | %.6f  |   %d   |  %d   |' % (p[0],p[1],predicted[i],y_test[i])

# examine the coefficients
print '---- Checking Coefs ----'
colNames.values[0] = 'Intercept'
coefDf = pd.DataFrame(zip(colNames, np.transpose(np.append(model.intercept_,model.coef_))))
print coefDf