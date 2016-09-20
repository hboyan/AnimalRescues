import matplotlib.pyplot as plt
import math

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cross_validation import train_test_split

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

df['adopted'] = df['outcome_type'] == "Adoption"
y = df['adopted'].astype(int)
y

X_train, X_test, y_train, y_test = train_test_split(X, y)

def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            if prob_pos.any > 1.0:
                prob_pos = 1

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

# Plot calibration cuve for Gaussian Naive Bayes
plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)

# Plot calibration cuve for Linear SVC
plot_calibration_curve(LinearSVC(), "SVC", 2)

plt.show()
