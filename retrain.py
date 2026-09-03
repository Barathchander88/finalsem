"""
Retrain and re-save model1.pkl.

Reproduces the pipeline from bank_Loan_Prediction.ipynb exactly — same
encodings, same imputation strategy, same train/test split — but under a
current scikit-learn, because the committed pickle was written with 0.24.1
and no longer loads.

One deliberate change: the notebook imputed some columns with
np.random.randint, which makes every training run produce a different model.
The seed below makes the result reproducible.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 0
np.random.seed(SEED)

# The order main.py builds its feature vector in. Must not drift.
FEATURES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train.Loan_Status.map({"Y": 1, "N": 0})
train = train.drop("Loan_Status", axis=1)
data = pd.concat([train, test], ignore_index=True)

data.Gender = data.Gender.map({"Male": 1, "Female": 0})
data.Married = data.Married.map({"Yes": 1, "No": 0})
data.Dependents = data.Dependents.map({"0": 0, "1": 1, "2": 2, "3+": 3})
data.Education = data.Education.map({"Graduate": 1, "Not Graduate": 0})
data.Self_Employed = data.Self_Employed.map({"Yes": 1, "No": 0})
data.Property_Area = data.Property_Area.map({"Urban": 2, "Rural": 0, "Semiurban": 1})

data["Credit_History"] = data.Credit_History.fillna(np.random.randint(0, 2))
data["Married"] = data.Married.fillna(np.random.randint(0, 2))
data["Gender"] = data.Gender.fillna(np.random.randint(0, 2))
data["Self_Employed"] = data.Self_Employed.fillna(np.random.randint(0, 2))
data["LoanAmount"] = data.LoanAmount.fillna(data.LoanAmount.median())
data["Loan_Amount_Term"] = data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean())
data["Dependents"] = data.Dependents.fillna(data.Dependents.median())

X_all = data[FEATURES]
X = X_all.iloc[:614]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=SEED)

candidates = [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
    ("Random Forest", RandomForestClassifier(random_state=SEED)),
    ("Decision Tree", DecisionTreeClassifier(random_state=SEED)),
    ("Support Vector Classifier", SVC(random_state=SEED)),
    ("K-Nearest Neighbours", KNeighborsClassifier()),
    ("Naive Bayes", GaussianNB()),
]

print(f"{'model':<30} {'holdout':>8} {'5-fold cv':>10}")
print("-" * 50)
rows = []
for name, clf in candidates:
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    cv = cross_val_score(clf, X, y, cv=5).mean()
    rows.append((name, acc, cv))
    print(f"{name:<30} {acc*100:>7.2f}% {cv*100:>9.2f}%")

# The notebook shipped Logistic Regression, so keep that choice — but wrap it
# in a scaler. Income is in thousands while every other feature is 0-3, and
# unscaled the solver never converges. A Pipeline is a drop-in: main.py still
# just calls model.predict(), so the app needs no change.
scaled = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scaled.fit(X_tr.values, y_tr)
scaled_acc = accuracy_score(y_te, scaled.predict(X_te.values))
print(f"\nLogistic Regression, scaled:   {scaled_acc*100:.2f}% holdout"
      f" ({cross_val_score(scaled, X.values, y, cv=5).mean()*100:.2f}% cv)")

# Majority class — the number any accuracy claim has to beat to mean anything.
majority = max(y_te.mean(), 1 - y_te.mean())
print(f"Always-predict-approved baseline: {majority*100:.2f}%")

# Fit on .values so no feature names are attached; the app predicts from a
# plain list, which otherwise warns on every prediction.
final = scaled
pred = final.predict(X_te.values)

print(f"\nShipped model: StandardScaler + Logistic Regression")
print(f"Holdout accuracy: {accuracy_score(y_te, pred)*100:.2f}%")
print(f"\nConfusion matrix (rows = actual, cols = predicted):\n{confusion_matrix(y_te, pred)}")
print(f"\n{classification_report(y_te, pred, target_names=['Rejected', 'Approved'])}")

with open("model1.pkl", "wb") as f:
    pickle.dump(final, f)
print("Saved model1.pkl")

# The exact row the notebook sanity-checked with.
check = [[0.0, 0.0, 0.0, 1, 0.0, 4230, 0.0, 112.0, 360.0, 1.0, 1]]
print("Sanity check row predicts:", final.predict(check))
