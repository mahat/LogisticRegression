'''
Author: mahat
'''

import pandas as pd
import numpy as np
from random import randrange

from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import Utils

# start with simple logistic regression
[X, Y,cols] = Utils.getTitanicDataSet()

# sperate training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# creating model with random regularization param
reg = pow(10, randrange(-4, 4))

model = linear_model.LogisticRegression(penalty='l2', C=reg)
model.fit(X_train, y_train)

# predict test set
preds = model.predict(X_test)
oldval = metrics.accuracy_score(y_test, preds)
print 'Default accuracy with random reg param %f' % oldval


# parameter tuning with using cross validation
# split train data into train and validation set
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3)

# creating folds
fold = KFold(n_splits=3, shuffle=True, random_state=42)

# Grid search
candidate_params = np.power(10.0, np.arange(-4, 4))
grid = {
    'C': candidate_params
    , 'solver': ['newton-cg']
}
clf = LogisticRegression(penalty='l2')
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=fold)
gs.fit(X_validation, y_validation)

print 'Grid Seach best score %.4f' % gs.best_score_
print 'Grid Seach Best regularization param: %f:' % candidate_params[gs.best_index_]

print '-------------------------------------------------------------------------------'
# or we can use LogisticRegressionCV in Sklearn
searchCV = LogisticRegressionCV(
    Cs=np.power(10.0, np.arange(-4, 4))
    , penalty='l2'
    , scoring='roc_auc'
    , cv=fold
    , solver='newton-cg'
    , refit=True
)

searchCV.fit(X_validation, y_validation)

print 'LogisticRegressionCV Max auc_roc: %f' % max(map(lambda x: x.mean(), searchCV.scores_[1]))
print 'LogisticRegressionCV Best regularization param: %f ' % searchCV.C_[0]

print '-------------------------------------------------------------------------------'
best_reg = candidate_params[gs.best_index_]
# checking model accuracy
scores = cross_val_score(LogisticRegression(C=best_reg, penalty='l2', solver='newton-cg'), X_train, y_train,
                         scoring='roc_auc', cv=5)

# print scores and its mean
print 'Scores: '
print scores
print 'Score means: '
print scores.mean()

# if it seems best regularization param we finds is stable so we can use it in training and testing

model = LogisticRegression(C=best_reg, penalty='l2', solver='newton-cg')
model.fit(X_train, y_train)

print '-------------------------- Prediction and Reporting -------------------------'
# testing model
predicted = model.predict(X_test)

# accuracy score
print 'old accuracy score %f' % oldval
print 'Accuracy score %f ' % metrics.accuracy_score(y_test, predicted)
# confusion matrix
print 'Confusion Matrix'
print metrics.confusion_matrix(y_test, predicted)
# classification report
print 'Classification report'
print metrics.classification_report(y_test, predicted)
