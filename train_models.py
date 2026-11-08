"""ML Assignment 2 - training script.

Trains 5 classifiers on one classification CSV dataset, evaluates each with
Accuracy, AUC, Precision, Recall, F1 and MCC, prints the comparison tables,
and saves the trained models plus a held-out test_data.csv for the Streamlit
app.

Usage:
    python train_models.py
The dataset location and layout are defined in the CONFIG block below.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = "model"
RANDOM_STATE = 42

# ----------------------------- CONFIG ---------------------------------------
# The breast_cancer_wisconsin.csv file has no header row: column 0 is the
# patient ID, column 1 is the diagnosis (M/B), columns 2..31 are features.
CSV_PATH = "breast_cancer_wisconsin.csv"
HAS_HEADER = False          # set True for a CSV whose first row is the header
TARGET_COLUMN = "col_1"     # column holding the class labels
DROP_COLUMNS = ["col_0"]    # non-feature columns such as row IDs
# ----------------------------------------------------------------------------


def build_models():
    """Return the five required classifiers."""
    return {
        # max_iter is raised so the solver converges on the scaled features.
        "Logistic Regression": LogisticRegression(max_iter=5000),
        # A fixed seed keeps the tree's split choices reproducible.
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        # Majority vote among the 5 nearest scaled neighbours.
        "KNN": KNeighborsClassifier(n_neighbors=5),
        # Gaussian likelihoods suit the continuous predictor values here.
        "Naive Bayes": GaussianNB(),
        # probability=True enables predict_proba, which the AUC metric needs.
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    }


def evaluate(model, X_test, y_test, average):
    """Compute the six required metrics for one fitted model."""
    y_pred = model.predict(X_test)
    # AUC needs scores, not hard labels; for binary data use class 1 probability.
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
        scores = scores[:, 1] if scores.shape[1] == 2 else scores
    else:
        scores = y_pred
    auc = roc_auc_score(y_test, scores, multi_class="ovr",
                        average=average if scores.ndim > 1 else None)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    if HAS_HEADER:
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.read_csv(CSV_PATH, header=None)
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    df = df.dropna(axis=1, how="all")          # drop empty trailing columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    if DROP_COLUMNS:
        df = df.drop(columns=DROP_COLUMNS)

    y = df[TARGET_COLUMN]
    if not pd.api.types.is_numeric_dtype(y):    # e.g. 'M'/'B' labels
        # LabelEncoder sorts B before M, making malignant (M) the positive class 1.
        y = pd.Series(LabelEncoder().fit_transform(y), index=df.index)
    X = df.drop(columns=[TARGET_COLUMN])
    # The configured dataset has 30 numeric predictors; exclude any text fields.
    X = X.select_dtypes(include=[np.number])    # keep numeric features only
    keep = X.dropna().index                     # use the same complete rows for X and y
    X, y = X.loc[keep], y.loc[keep]

    n_classes = len(np.unique(y))
    average = "binary" if n_classes == 2 else "weighted"
    print(f"Dataset: {X.shape[0]} instances, {X.shape[1]} features, "
          f"{n_classes} classes")

    # Stratification preserves the class balance in both splits; the fixed seed
    # makes the reported metrics and generated test_data.csv reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Fit scaling on training rows only to avoid leaking test-set information.
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Save the preprocessing artifacts alongside each fitted model for app.py.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as fh:
        json.dump(list(X.columns), fh)

    # Evaluate and persist every classifier using the same scaled train/test data.
    rows = []
    for name, model in build_models().items():
        model.fit(X_train_s, y_train)
        metrics = evaluate(model, X_test_s, y_test, average)
        rows.append({"ML Model Name": name, **metrics})
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, filename))
        print(f"saved {filename}")

    # Round only for display; the saved models keep full precision.
    table = pd.DataFrame(rows).round(4)
    print("\n=== Model comparison on the held-out test split ===")
    print(table.to_string(index=False))

    # F1 balances precision and recall, which suits this imbalanced dataset.
    best = table.loc[table["F1"].idxmax(), "ML Model Name"]
    print(f"\nBest model by F1 on this test split: {best}")

    # Keep the true labels with the held-out features so app.py can calculate metrics.
    test_out = X_test.copy()
    test_out[TARGET_COLUMN] = y_test
    test_out.to_csv("test_data.csv", index=False)
    print("test_data.csv written (features + target column).")


if __name__ == "__main__":
    main()