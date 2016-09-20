from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
import pandas as pd

features = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/dog_dummies_feats.csv')
features.columns
targets = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/dog_dummies_targets.csv')
targets.columns
X = features.copy()
y = targets
X.drop('Unnamed: 0', inplace=True, axis=1)
X.drop('condition_in', inplace=True, axis=1)
X.drop('intake_type', inplace=True, axis=1)
X.drop('outcome_detail', inplace=True, axis=1)
X.drop('outcome_type', inplace=True, axis=1)
X.drop('breeds', inplace=True, axis=1)
X.drop('colors', inplace=True, axis=1)
X.drop('groups', inplace=True, axis=1)
X.drop('intake_time', inplace=True, axis=1)
X.drop('outcome_time', inplace=True, axis=1)
X.set_index('id', inplace=True)
X.dtypes

def predicto(y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #SVMs are good for multifeature (100+ data points)
    #tune gamma and C
    #gamma is a parameter for rbf
        #high gamma: low bias = correct a lot, high variance = captures more of the data but only the training data
        #low gamma: high bias = wrong a lot, low variance = less erratic
    #C is a penalty parameter for the error term
        #small c: decision surface is smooth and simple (but can underfit - high bias/low variance)
        #big c: jagged margin (can overfit - low bias/high variance)
    #to test - use GridSearchCV:
        #gamma_range = 10.**np.arange(-5,2)
        #C_range = 10.**np.arange(-2,3)
        #kernel_range = ['rbf', 'sigmoid', 'linear', 'poly']
        #param_grid=dict(gamma=gamma_range, c=C_range, kernel=kernel_range),
        #grid=GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
        #grid.fit(x,y)
        #this fits 1400 SVMs (7 gammas, 5 c values, 4 kernels, 10 cross folds)
    #look at results:
        #plot accuracy, this represents bias (y axis) vs gamma (x axis, representing complexity)
        #need to find optimum model complexity - we plot accuracy, which is inverse error - so peak is ideal complexity
        #grid.best_score_, grid.best_params_, grid.best_estimator_
    #roc_auc only works for binary classification (needs true positive rate)
    model = SVC(C = .01, )
    model.fit(X_train, y_train)
    coefs = pd.DataFrame([list(model.coef_[0]), list(X.columns)]).transpose()
    print model.score(X,y)
    print cross_val_score(model, X,y,cv=3).mean()
    coefs = coefs.sort_values(0, ascending = False)

    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    #eval with cross_val_score().mean()
    print 'Confusion Matrix: '
    print confusion_matrix(y_test, y_pred)
    print 'Acc score: '
    print accuracy_score(y_test, y_pred)
    print 'Classification Report: '
    print classification_report(y_test, y_pred)
    print coefs[coefs[0] != 0]

# X_train, X_test, y_train, y_test = train_test_split(X, y)

# targets.columns
# for item in targets.columns[:9]:
#     y = targets[item]
#     print item
#     if y.sum() > 200:
#         predicto(y)
#     else:
#         print "too few samples"

predicto(y['outcome_type'])
