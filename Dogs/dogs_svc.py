import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

df = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dogs_tomodel.csv')

X = df[['age_in',
 'age_out',
 'fixed_in',
 'fixed_out',
 'fixed_in_care',
 'time_in_care',
 'male',
 'mix',
 'White',
 'Tan',
 'Blue',
 'Black',
 'Buff',
 'Silver',
 'Tricolor',
 'Brown',
 'Red',
 'Gray',
 'Brown Brindle',
 'Cream',
 'Sable',
 'Fawn',
 'Chocolate',
 'Blue Merle',
 'Yellow',
 'Black Brindle',
 'FSS',
 'Working Group',
 'Terrier Group',
 'Herding Group',
 'Sporting Group',
 'Hound Group',
 'Non-Sporting Group',
 'Toy Group',
 'Bully Breed',
 'Hound',
 'intake_condition_Aged',
 'intake_condition_Feral',
 'intake_condition_Injured',
 'intake_condition_Normal',
 'intake_condition_Nursing',
 'intake_condition_Other',
 'intake_condition_Pregnant',
 'intake_condition_Sick',
 'intake_type_Euthanasia Request',
 'intake_type_Owner Surrender',
 'intake_type_Public Assist',
 'intake_type_Stray']]

from sklearn.preprocessing import LabelEncoder
def groupminis(x):
    if str(x) in ['Died','Disposal','Missing']:
        return 'Other'
    else:
        return x
df['outcome_type'] = df['outcome_type'].apply(groupminis)
le = LabelEncoder()
df['outcome_labels'] = le.fit_transform(df['outcome_type'])
outcome_labels = le.classes_
outcome_labels
y = df['outcome_labels']


# to test - use GridSearchCV:
# model = SVC()
# kernel_range = ['rbf', 'sigmoid', 'linear', 'poly'] #linear was so vastly better, just switched to linearSVC
'''
gridsearch output:
Fitting 3 folds for each of 140 candidates, totalling 420 fits
[CV] kernel=rbf, C=0.01, gamma=1e-05 .................................
[CV] ........ kernel=rbf, C=0.01, gamma=1e-05, score=0.426866 -  30.5s
[CV] kernel=rbf, C=0.01, gamma=1e-05 .................................
[CV] ........ kernel=rbf, C=0.01, gamma=1e-05, score=0.426908 -  30.6s
[CV] kernel=rbf, C=0.01, gamma=1e-05 .................................
[CV] ........ kernel=rbf, C=0.01, gamma=1e-05, score=0.426971 -  30.1s
[CV] kernel=sigmoid, C=0.01, gamma=1e-05 .............................
[CV] .... kernel=sigmoid, C=0.01, gamma=1e-05, score=0.426866 -  27.5s
[CV] kernel=sigmoid, C=0.01, gamma=1e-05 .............................
[CV] .... kernel=sigmoid, C=0.01, gamma=1e-05, score=0.426908 -  27.1s
[CV] kernel=sigmoid, C=0.01, gamma=1e-05 .............................
[CV] .... kernel=sigmoid, C=0.01, gamma=1e-05, score=0.426971 -  27.2s
[CV] kernel=linear, C=0.01, gamma=1e-05 ..............................
[CV] ..... kernel=linear, C=0.01, gamma=1e-05, score=0.924030 -   8.2s
[CV] kernel=linear, C=0.01, gamma=1e-05 ..............................
[CV] ..... kernel=linear, C=0.01, gamma=1e-05, score=0.893385 -   8.1s
[CV] kernel=linear, C=0.01, gamma=1e-05 ..............................
[CV] ..... kernel=linear, C=0.01, gamma=1e-05, score=0.901284 -   7.9s
[CV] kernel=poly, C=0.01, gamma=1e-05 ................................
[CV] ....... kernel=poly, C=0.01, gamma=1e-05, score=0.426866 -  26.2s
[CV] kernel=poly, C=0.01, gamma=1e-05 ................................
[CV] ....... kernel=poly, C=0.01, gamma=1e-05, score=0.426908 -  26.8s
[CV] kernel=poly, C=0.01, gamma=1e-05 ................................
[CV] ....... kernel=poly, C=0.01, gamma=1e-05, score=0.426971 -  26.5s
[CV] kernel=rbf, C=0.01, gamma=0.0001 ................................
[CV] ....... kernel=rbf, C=0.01, gamma=0.0001, score=0.426866 -  32.9s
[CV] kernel=rbf, C=0.01, gamma=0.0001 ................................
[CV] ....... kernel=rbf, C=0.01, gamma=0.0001, score=0.426908 -  33.3s
[CV] kernel=rbf, C=0.01, gamma=0.0001 ................................
[CV] ....... kernel=rbf, C=0.01, gamma=0.0001, score=0.426971 -  32.1s
[CV] kernel=sigmoid, C=0.01, gamma=0.0001 ............................
[CV] ... kernel=sigmoid, C=0.01, gamma=0.0001, score=0.426866 -  32.3s
[CV] kernel=sigmoid, C=0.01, gamma=0.0001 ............................
[CV] ... kernel=sigmoid, C=0.01, gamma=0.0001, score=0.424817 -  29.4s
[CV] kernel=sigmoid, C=0.01, gamma=0.0001 ............................
[CV] ... kernel=sigmoid, C=0.01, gamma=0.0001, score=0.426822 -  30.2s
[CV] kernel=linear, C=0.01, gamma=0.0001 .............................
[CV] .... kernel=linear, C=0.01, gamma=0.0001, score=0.924030 -   9.0s
[CV] kernel=linear, C=0.01, gamma=0.0001 .............................
[CV] .... kernel=linear, C=0.01, gamma=0.0001, score=0.893385 -   8.0s
[CV] kernel=linear, C=0.01, gamma=0.0001 .............................
[CV] .... kernel=linear, C=0.01, gamma=0.0001, score=0.901284 -   8.2s
[CV] kernel=poly, C=0.01, gamma=0.0001 ...............................
[CV] ...... kernel=poly, C=0.01, gamma=0.0001, score=0.426866 -  28.6s
[CV] kernel=poly, C=0.01, gamma=0.0001 ...............................
[Parallel(n_jobs=1)]: Done   1 tasks       | elapsed:   30.5s
[Parallel(n_jobs=1)]: Done   4 tasks       | elapsed:  2.0min
[Parallel(n_jobs=1)]: Done   7 tasks       | elapsed:  3.0min
[Parallel(n_jobs=1)]: Done  12 tasks       | elapsed:  4.6min
[Parallel(n_jobs=1)]: Done  17 tasks       | elapsed:  7.3min
'''

#tune params
# model = LinearSVC(loss='squared_hinge', dual=False) #dual has to be false to play nice with both penalty types - if dual=True can't use l1 and loss='squared_hinge'
# C_range = 10.**np.arange(-2,5) #best C is 10.0, with score of 0.69218135669138503
# penalty = ['l1','l2'] #consistently l2, with score of 0.69218135669138503
# param_grid=dict(penalty=penalty, C=C_range)
# grid=GridSearchCV(model, param_grid, cv=10, scoring='accuracy', verbose=10)
# grid.fit(X,y)
#
# grid.best_estimator_
# grid.best_params_
# grid.best_score_


#double check loss and dual combos
# model = LinearSVC(C=100.0, penalty='l2', loss='squared_hinge', dual=False)
# model.fit(X,y)
# cross_val_score(model, X, y, cv=12).mean()
#
# model = LinearSVC(C=100.0, penalty='l2', loss='hinge', dual=True)
# model.fit(X,y)
# cross_val_score(model, X, y, cv=12).mean()
#
# model = LinearSVC(C=100.0, penalty='l1', loss='squared_hinge', dual=False)
# model.fit(X,y)
# cross_val_score(model, X, y, cv=12).mean()
#confirms loss='squared_hinge' and dual=False

bestmodel = LinearSVC(loss='squared_hinge', dual=False, C=100.0, penalty='l2')
bestmodel.fit(X,y)
cross_val_score(bestmodel, X, y, cv=12).mean() #0.69188207476882313

#interpret it
coefs = pd.DataFrame(X.columns, [bestmodel.coef_.mean(axis=0)])
coefs.reset_index(inplace=True)
coefs.rename(columns={'index':'coef',0:'feature'}, inplace=True)
coefs.sort_values('coef').tail()


#roc_auc only works for binary classification (needs true positive rate)
#     model = SVC(C = .01, )
#     model.fit(X_train, y_train)
#     coefs = pd.DataFrame([list(model.coef_[0]), list(X.columns)]).transpose()
#     print model.score(X,y)
#     print cross_val_score(model, X,y,cv=3).mean()
#     coefs = coefs.sort_values(0, ascending = False)
#
#     model.fit(X_train, y_train)
#     model.score(X_test, y_test)
#     y_pred = model.predict(X_test)
#
#     #eval with cross_val_score().mean()
#     print 'Confusion Matrix: '
#     print confusion_matrix(y_test, y_pred)
#     print 'Acc score: '
#     print accuracy_score(y_test, y_pred)
#     print 'Classification Report: '
#     print classification_report(y_test, y_pred)
#     print coefs[coefs[0] != 0]
#
# # X_train, X_test, y_train, y_test = train_test_split(X, y)
#
# # targets.columns
# # for item in targets.columns[:9]:
# #     y = targets[item]
# #     print item
# #     if y.sum() > 200:
# #         predicto(y)
# #     else:
# #         print "too few samples"
#
# predicto(y['outcome_type'])
