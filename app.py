"""ML Assignment 2 - Streamlit app.

Loads the five saved classifiers, lets the user upload a test CSV
(features + target column), pick a model, and view evaluation metrics,
a confusion matrix and a classification report - all without retraining.

Run locally:  streamlit run app.py
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="ML Assignment 2 - Classifier Comparison",
                   layout="wide")
st.title("Classification Model Comparator")
st.write("Upload a test CSV (features + target), choose one of the five "
         "trained classifiers, and inspect its evaluation metrics.")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as fh:
        feature_names = json.load(fh)
    models = {name: joblib.load(os.path.join(MODEL_DIR, fname))
              for name, fname in MODEL_FILES.items()}
    return scaler, feature_names, models


scaler, feature_names, models = load_artifacts()

# --- (a) Dataset upload option ------------------------------------------------
uploaded = st.file_uploader("Upload test data (CSV with features + target)",
                            type=["csv"])
target_col = st.text_input("Target column name", value="col_1")

if uploaded is None:
    st.info("Waiting for a CSV upload. Use the test_data.csv produced by "
            "train_models.py.")
    st.stop()

data = pd.read_csv(uploaded)
if target_col not in data.columns:
    st.error(f"Target column '{target_col}' not found in the uploaded CSV.")
    st.stop()

missing = [c for c in feature_names if c not in data.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

X = data[feature_names]
y_raw = data[target_col]
# Map string labels (e.g. M/B) to the integer encoding used in training.
if not pd.api.types.is_numeric_dtype(y_raw):
    classes = sorted(y_raw.unique())
    y = y_raw.map({c: i for i, c in enumerate(classes)}).to_numpy()
else:
    y = y_raw.to_numpy()
X_scaled = scaler.transform(X)

# --- (b) Model selection dropdown ---------------------------------------------
model_name = st.selectbox("Select a model", list(MODEL_FILES.keys()))
model = models[model_name]

y_pred = model.predict(X_scaled)
n_classes = len(np.unique(y))
average = "binary" if n_classes == 2 else "weighted"

# --- (c) Evaluation metrics ----------------------------------------------------
if hasattr(model, "predict_proba"):
    scores = model.predict_proba(X_scaled)
    scores = scores[:, 1] if scores.shape[1] == 2 else scores
else:
    scores = y_pred
auc = roc_auc_score(y, scores, multi_class="ovr",
                    average=average if np.ndim(scores) > 1 else None)
metrics = {
    "Accuracy": accuracy_score(y, y_pred),
    "AUC": auc,
    "Precision": precision_score(y, y_pred, average=average, zero_division=0),
    "Recall": recall_score(y, y_pred, average=average, zero_division=0),
    "F1 Score": f1_score(y, y_pred, average=average, zero_division=0),
    "MCC": matthews_corrcoef(y, y_pred),
}

st.subheader(f"Evaluation metrics - {model_name}")
cols = st.columns(len(metrics))
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

# --- (d) Confusion matrix + classification report ------------------------------
st.subheader("Confusion matrix")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks(range(cm.shape[1]))
ax.set_yticks(range(cm.shape[0]))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")
fig.colorbar(im, ax=ax)
st.pyplot(fig)

st.subheader("Classification report")
st.text(classification_report(y, y_pred, zero_division=0))
