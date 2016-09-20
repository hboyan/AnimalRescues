import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/OtherAnimals/other_tomodel.csv')
df.breed = df.breed.apply(lambda x: str(x).replace(' Mix', ''))

#format features
condition_dummies = pd.get_dummies(df.condition_in)
intake_dummies = pd.get_dummies(df.intake_type)

breeds = dict(df.breed.value_counts())
def breedshrink(x):
    if breeds[x] < 50:
        return 'Other'
    else:
        return x
df.breed = df.breed.apply(breedshrink)
breed_dummies = pd.get_dummies(df.breed)

X = pd.concat([intake_dummies, condition_dummies, breed_dummies], axis=1)

#format target
df.groupby('outcome_type').outcome_detail.value_counts()
df['real_outcome'] = df['outcome_type'] + '_' + df['outcome_detail']
df[['outcome_detail', 'outcome_type', 'real_outcome']]
def finaloutcome(x):
    if x == 'Euthanasia_Rabies Risk':
        return 'Euthanasia_Rabies'
    else:
        return x.split('_')[0]
df['outcome'] = df.real_outcome.apply(finaloutcome)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df.outcome)
y_names = le.classes_

#set goal - null model is 50% Euthanasia_Rabies
df.outcome.value_counts() / len(df)

#build the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
model = DecisionTreeClassifier()

#evaluate the model
model.fit(X,y)
cross_val_score(model, X, y, cv=10).mean()

from sklearn.grid_search import GridSearchCV
param_grid = {
    'criterion':('gini', 'entropy'),
    'splitter':('best', 'random'),
    'max_depth': range(5, 20),
    'max_features': range(1, X.shape[1]),
    'min_samples_leaf': range(1, 5),
    'min_samples_split': range(1, 5),
    'presort': ('True', 'False')
    }
gs = GridSearchCV(model, param_grid, verbose=1)
# gs.fit(X,y)
# gs.best_params_

best_model = DecisionTreeClassifier(max_depth=10, max_features=2, presort=True)
best_model.fit(X,y)
cross_val_score(best_model, X, y, cv=10).mean()

# try another model - random forest
rf = RandomForestClassifier()
rf.fit(X,y)
cross_val_score(rf, X, y, cv=10).mean()

param_grid = {
    'n_estimators':range(1,20),
    'max_depth': range(5, 20),
    'min_samples_leaf': range(1, 5),
    'min_samples_split': range(1, 5),
    }
rfgs = GridSearchCV(rf, param_grid, verbose=1)
rfgs.fit(X,y)
rfgs.best_params_
best_rf = RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=4, n_estimators=8)
cross_val_score(best_rf, X, y, cv=10).mean()

# Result - all the models are right around 70% accurate.

#visualize the model
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(best_model, out_file=dot_data,
                         feature_names=X.columns,
                         class_names=y_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
