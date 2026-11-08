# ML Assignment 2 - Classification Models with Streamlit Deployment

**BITS ID:** 2026AK05394

## a. Problem Statement

This project compares five supervised classifiers for predicting whether a
breast mass is benign or malignant from measurements computed from a digitized
fine needle aspirate image. The models are evaluated on a held-out stratified
test split using Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation
Coefficient (MCC). A Streamlit app loads saved models and evaluates an uploaded
labeled CSV without retraining. This is an educational exercise, not a medical
diagnostic tool.

## b. Dataset Description

The Wisconsin Diagnostic Breast Cancer dataset is from the UCI Machine
Learning Repository. It contains measurements computed from digitized images
of fine needle aspirates of breast masses. The 30 numeric predictors describe
the mean, standard error, and largest ("worst") values of ten cell-nucleus
characteristics: radius, texture, perimeter, area, smoothness, compactness,
concavity, concave points, symmetry, and fractal dimension.

- Source: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Instances: 569 | Features: 30 numeric predictors | Classes: 2
- Class counts: 357 benign (B), 212 malignant (M)
- Target: diagnosis, encoded as B = 0 and M = 1 for model training
- The raw CSV has no header row. Column 0 is an ID and is excluded; column 1
	is the diagnosis; columns 2-31 are the 30 predictors.

## c. GitHub Repository Link

GitHub repository URL: Not provided yet.

Live Streamlit app URL: Not deployed yet.

## d. Models Used

The dataset was divided using a stratified 80/20 train/test split with
`random_state=42`. The scaler was fitted only on the training data and then
used to transform both splits. All five classifiers use the same split. The
metrics below are from the 114-row held-out test split. Precision, Recall, and
F1 use malignant (M) as the positive class; AUC uses predicted probabilities.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong overall performance and the highest AUC (0.9960), indicating excellent class ranking on this split. |
| Decision Tree | Most interpretable model, but it has the lowest AUC (0.9246), F1 (0.9048), and MCC (0.8492) on this split. |
| KNN | Performs well overall (accuracy 0.9561); its lower recall (0.9048) means it misses more malignant cases than Logistic Regression and SVM on this split. |
| Naive Bayes | Has a high AUC (0.9891), but lower recall (0.8571) and accuracy (0.9211) at its default decision rule. |
| Support Vector Machine | Best accuracy (0.9737), precision (1.0000), F1 (0.9630), and MCC (0.9442); its AUC (0.9947) is slightly below Logistic Regression. |

**Overall winner for this dataset:** Support Vector Machine based on the
highest F1, accuracy, precision, and MCC on the held-out split. Logistic
Regression has the highest AUC, so the winner depends somewhat on which metric
is prioritized. These results come from one split and do not guarantee
performance on new clinical data.

## Running the project

Install the packages and make sure `breast_cancer_wisconsin.csv` is in the
project directory. Then run:

```bash
python3 train_models.py
python3 -m streamlit run app.py
```

The training script writes the five fitted models, scaler, feature list, and
the held-out labeled `test_data.csv`. In the app, upload `test_data.csv` and
use `col_1` as the target column.

## Repository structure

```
├── breast_cancer_wisconsin.csv # source dataset
├── app.py              # Streamlit app (loads saved models, no retraining)
├── train_models.py     # training + evaluation source code
├── requirements.txt
├── README.md
├── test_data.csv       # held-out test split with target column
└── model/              # saved .joblib models (all five) + scaler + feature names
```