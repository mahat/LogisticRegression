import numpy as np
from sklearn import datasets, linear_model, metrics

# start with simple logistic regression
# loading data
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
# sperate training and test
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
# creating model
reg = 1
model = linear_model.LogisticRegression(C=reg)
model.fit(X_train, y_train)

# predict test set
preds = model.predict(X_test)
print metrics.accuracy_score(y_test, preds)

#TODO: kod duzenlemesi

# determine best regularization using cross validation
# split train data into train and validation set
X_train, X_test, X_validation, y_validation = train_test_split(X_train, y_train, test_size=0.3)
# Grid search

fold = KFold(len(y), n_folds=5, shuffle=True, random_state=777)

grid = {
    'C': np.power(10.0, np.arange(-10, 10))
    , 'solver': ['newton-cg']
}
clf = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=fold)
gs.fit(X_validation, y_validation)

print 'Grid Seach best score %.4f' % gs.best_score_

# or we can use LogisticRegressionCV in Sklearn
searchCV = LogisticRegressionCV(
        Cs=list(np.power(10, np.arange(-4, 4)))
        ,penalty='l2'
        ,scoring='roc_auc'
        ,cv=fold
        ,random_state=777
        ,max_iter=10000
        ,fit_intercept=True
        ,solver='newton-cg'
        ,tol=10
    )

searchCV.fit(X_validation, y_validation)

print ('Max auc_roc:', searchCV.scores_[1].max())


# checking model accuracy
scores = cross_val_score(searchCV(), X_validation, y_validation, scoring='accuracy', cv=10)

# print scores and its mean
print scores
print scores.mean()