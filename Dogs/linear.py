from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd
from sklearn.cross_validation import cross_val_score

df = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dogs_tomodel.csv')

X = df[['age_in',
 'age_out',
 'fixed_in',
 'fixed_out',
 'fixed_in_care',
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

y = df['time_in_care']

# from sklearn.preprocessing import LabelEncoder
# def groupminis(x):
#     if str(x) in ['Died','Disposal','Missing']:
#         return 'Other'
#     else:
#         return x
# df['outcome_type'] = df['outcome_type'].apply(groupminis)
# le = LabelEncoder()
# df['outcome_labels'] = le.fit_transform(df['outcome_type'])
# outcome_labels = le.classes_
# outcome_labels
# y = df['outcome_labels']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
model.score(X,y)

# solvers = ['newton-cg', 'lbfgs']
# C_range = 10.**np.arange(-2,5) #best C is 10.0, with score of 0.69218135669138503
# param_grid=dict(solver=solvers, C=C_range)
# grid=GridSearchCV(model, param_grid, cv=10, scoring='accuracy', verbose=10)
# grid.fit(X,y)
