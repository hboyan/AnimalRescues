from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.cross_validation import cross_val_score, train_test_split
from datetime import timedelta, datetime
import numpy as np
import patsy as pt


import pandas as pd
df1 = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/AustinTake2/timelines.csv')
df2 = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/AustinTake2/animals.csv')
df = df1.merge(df2, how='left', on='id')
df.drop('Unnamed: 0_x', axis=1, inplace=True)
df.drop('Unnamed: 0_y', axis=1, inplace=True)
df.drop('id_check', axis=1, inplace=True)
df['intake_time'] = pd.to_datetime(df['intake_time'])
df['outcome_time'] = pd.to_datetime(df['outcome_time'])
df['fixed_in_care'] = df.fixed_in != df.fixed_out
df['fixed_in_care'] = df['fixed_in_care'].astype('int')
df['time_held'] = (df['outcome_time'] - df['intake_time'])
df['male'] = df['male'].convert_objects(convert_numeric=True)
helper = np.vectorize(lambda x: x.total_seconds())
df['time_held'] = df['time_held'].apply(helper)

df.dtypes
y, X = pt.dmatrices('outcome_type ~ age_in + age_out + C(condition_in) + fixed_in + fixed_out + C(intake_type) + C(breed) + C(color) + male + C(species) + fixed_in_care + time_held', df, return_type='dataframe')

lm = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
X
y.describe()

for col in y:
    X_train, X_test, y_train, y_test = train_test_split(X, y[col])
    print col
    try:
        lm.fit(X_train, y_train)
        y_pred = lm.predict(X_test)
        y_pred_proba = lm.predict_proba(X_test)

        print 'classification report: '
        try:
            print classification_report(y_test, y_pred)
        except:
            print 'error'

        print 'confusion matrix: '
        try:
            print confusion_matrix(y_test, y_pred)
        except:
            'error'

        print 'accuracy score: '
        try:
            print accuracy_score(y_test, y_pred)
        except:
            print 'error'

        print 'roc auc score: '
        try:
            print roc_auc_score(y_test, y_pred_proba)
        except:
            print 'error'

        print 'cross val score: '
        try:
            print cross_val_score(lm, X_train, y_train)
        except:
            print 'error'

        print
        print 'coefficents: '
        try:
            df_coefs = pd.DataFrame(zip(list(X.columns), list(lm.coef_[0])), columns=['feature','coef']).sort_values('coef', ascending=False)[:5]
            print df_coefs
        except:
            print 'error'
    except:
        print 'Not enough classes in this data'

'''
Outputs:

outcome_type[Adoption]
classification report:
             precision    recall  f1-score   support

        0.0       0.76      0.73      0.75      4813
        1.0       0.64      0.67      0.65      3390

avg / total       0.71      0.71      0.71      8203
confusion matrix:
[[3536 1277]
 [1118 2272]]
accuracy score:
0.708033646227
roc auc score:
error
cross val score:
[ 0.7053517   0.71193466  0.70839937]
coefficents:
                                            feature          coef
2100                                      time_held  2.053112e-07
2099                                  fixed_in_care  5.022887e-08
689   C(breed)[T.Dachshund Mix Chihuahua Shorthair]  2.538942e-10
1140             C(breed)[T.Miniature Pinscher Mix]  2.098765e-10
1546    C(breed)[T.Soft Coated Wheaten Terrier Mix]  1.728996e-10

outcome_type[Died]
classification report:
             precision    recall  f1-score   support

        0.0       0.99      1.00      1.00      8148
        1.0       0.00      0.00      0.00        55

avg / total       0.99      0.99      0.99      8203
confusion matrix:
[[8145    3]
 [  55    0]]
accuracy score:
0.992929416067
roc auc score:
error
cross val score:
[ 0.99317406  0.99427039  0.99439161]
coefficents:
                                           feature      coef
2094                                        age_in  0.318612
2070                        C(color)[T.White Pink]  0.000165
761     C(breed)[T.Domestic Shorthair Mix Siamese]  0.000164
947                          C(breed)[T.Himalayan]  0.000144
616   C(breed)[T.Chow Chow Mix Labrador Retriever]  0.000143

outcome_type[Disposal]
classification report:
             precision    recall  f1-score   support

        0.0       1.00      1.00      1.00      8202
        1.0       0.00      0.00      0.00         1

avg / total       1.00      1.00      1.00      8203
confusion matrix:
[[8202    0]
 [   1    0]]
accuracy score:
0.99987809338
roc auc score:
error
cross val score:
[ 0.99975622  0.99975619  0.99987808]
coefficents:
                                            feature      coef
41                 C(breed)[T.American Bulldog Mix]  0.008777
563   C(breed)[T.Chihuahua Shorthair Mix Dachshund]  0.008759
1964                          C(color)[T.Red White]  0.008431
1406                            C(breed)[T.Raccoon]  0.008359
1534                          C(breed)[T.Skunk Mix]  0.007932

outcome_type[Euthanasia]
classification report:
             precision    recall  f1-score   support

        0.0       0.94      1.00      0.97      7687
        1.0       0.00      0.00      0.00       516

avg / total       0.88      0.94      0.91      8203
confusion matrix:
[[7687    0]
 [ 516    0]]
accuracy score:
0.937096184323
roc auc score:
error
cross val score:
[ 0.9407606   0.94075338  0.94086808]
coefficents:
                         feature          coef
1     C(condition_in)[T.Injured]  8.799245e-09
6        C(condition_in)[T.Sick]  1.478688e-09
10    C(intake_type)[T.Wildlife]  1.355814e-09
1221     C(breed)[T.Opossum Mix]  3.464009e-10
1407     C(breed)[T.Raccoon Mix]  2.525805e-10

outcome_type[Missing]
classification report:
             precision    recall  f1-score   support

        0.0       1.00      1.00      1.00      8197
        1.0       0.00      0.00      0.00         6

avg / total       1.00      1.00      1.00      8203
confusion matrix:
[[8197    0]
 [   6    0]]
accuracy score:
0.999268560283
roc auc score:
error
cross val score:
[ 0.99939054  0.99951237  0.99951231]
coefficents:
                                                feature          coef
1676      C(breed)[T.Yorkshire Terrier Mix Rat Terrier]  8.453896e-08
2083                         C(color)[T.Yellow Brindle]  8.132459e-08
247                       C(breed)[T.Beagle Mix Vizsla]  4.055738e-10
1121  C(breed)[T.Manchester Terrier Mix Norfolk Terr...  2.363852e-10
1245                              C(breed)[T.Pekingese]  1.852965e-10

outcome_type[Relocate]
classification report:
             precision    recall  f1-score   support

        0.0       1.00      1.00      1.00      8203
        1.0       0.00      0.00      0.00         0

avg / total       1.00      1.00      1.00      8203
confusion matrix:
[[8202    1]
 [   0    0]]
accuracy score:
0.99987809338
roc auc score:
error
cross val score:
error
coefficents:
                         feature      coef
1     C(condition_in)[T.Injured]  0.001875
10    C(intake_type)[T.Wildlife]  0.001872
2093         C(species)[T.Other]  0.001872
1220         C(breed)[T.Opossum]  0.001864
1825     C(color)[T.Brown White]  0.001864

outcome_type[Return to Owner]
classification report:
             precision    recall  f1-score   support

        0.0       0.87      0.93      0.90      6578
        1.0       0.60      0.43      0.50      1625

avg / total       0.81      0.83      0.82      8203
confusion matrix:
[[6112  466]
 [ 932  693]]
accuracy score:
0.829574545898
roc auc score:
error
cross val score:
[ 0.82482019  0.82250396  0.82555163]
coefficents:
                              feature          coef
2095                          age_out  7.008582e-05
2094                           age_in  6.987420e-05
2096                         fixed_in  3.884679e-08
8     C(intake_type)[T.Public Assist]  2.965042e-08
2091                C(species)[T.Dog]  2.325575e-08

outcome_type[Transfer]
classification report:
             precision    recall  f1-score   support

        0.0       0.74      0.98      0.84      5603
        1.0       0.85      0.26      0.40      2600

avg / total       0.78      0.75      0.70      8203
confusion matrix:
[[5483  120]
 [1916  684]]
accuracy score:
0.751798122638
roc auc score:
error
cross val score:
[ 0.6803998   0.68036084  0.68044379]
coefficents:
                                 feature      coef
2090                   C(species)[T.Cat]  0.107230
759   C(breed)[T.Domestic Shorthair Mix]  0.091365
3             C(condition_in)[T.Nursing]  0.018029
1811             C(color)[T.Brown Tabby]  0.016056
1685                   C(color)[T.Black]  0.009336


 '''

df.breed.value_counts()
