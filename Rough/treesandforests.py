from sklearn.tree import DecisionTreeClassifier
X = mappydf[['age_in','age_out','condition_in','fixed_in','fixed_out','intake_type','male','species','fixed_in_care']]
y = mappydf[['outcome_type']]
clf = DecisionTreeClassifier()

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y)
clf= clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print classification_report(y_test, y_pred)

print accuracy_score(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print accuracy_score(y_test, y_pred)
