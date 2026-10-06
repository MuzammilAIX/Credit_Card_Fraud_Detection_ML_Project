Credit Card Fraud Detection

An end-to-end machine learning project for detecting fraudulent credit card transactions using a tuned XGBoost classifier. The project focuses on imbalanced classification, fraud-focused evaluation metrics, feature engineering, hyperparameter tuning, model persistence, and Streamlit deployment.

Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent a very small portion of the dataset.

The objective is to build a model that can identify fraudulent transactions while maintaining strong precision and recall.

Dataset

The project uses the standard Credit Card Fraud Detection dataset with:

Time — transaction time

V1 to V28 — PCA-transformed transaction features

Amount — transaction amount

Class — target variable

0 = Legitimate

1 = Fraudulent

Data preparation included duplicate removal and analysis of the class imbalance.

Machine Learning Workflow

Data understanding and quality checks

Duplicate removal

Class distribution analysis

Train-test split with stratification

Feature engineering

Extracted Hour from Time

Created Amount_log using log1p

Feature scaling using StandardScaler

Model comparison

Logistic Regression

SMOTE-based approach

Decision Tree

Random Forest

XGBoost

Hyperparameter tuning with GridSearchCV

Evaluation using fraud-focused metrics

Model and scaler serialization

Streamlit deployment

Final Model

The deployed model is a tuned XGBoost classifier.

Best hyperparameters:

n_estimators: 200
max_depth: 7
learning_rate: 0.1
subsample: 1.0
colsample_bytree: 0.8

Evaluation on the test set:

Metric

Score

Precision

0.9610

Recall

0.7789

F1-Score

0.8605

ROC-AUC

0.9805

PR-AUC

0.8246

Confusion Matrix:

[[56648,     3],
 [   21,    74]]

The model uses a decision threshold of 0.50 for the deployed application.

Feature Importance

The most influential features in the final XGBoost model included:

V14

V10

V4

V12

Amount_log

V3

V18

V19

V8

V11

Because V1 to V28 are PCA-transformed features, they do not have direct business interpretations such as customer age or merchant type.

Project Structure

Credit_Card_Fraud_Detection_ML_Project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── credit_card.csv
│
├── models/
│   ├── xgb_fraud_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── preprocessing.py
│   └── train.py
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
│
├── requirements.txt
└── README.md

Run Locally

1. Clone the repository

git clone <your-repository-url>
cd Credit_Card_Fraud_Detection_ML_Project

2. Create and activate a virtual environment

Windows PowerShell:

python -m venv .venv
.venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Train the model

python src/train.py

The trained model and scaler are saved in the models/ directory.

5. Start the Streamlit application

python -m streamlit run app/app.py

The application provides a web interface for entering transaction information and receiving a fraud probability and classification.

Deployment

The Streamlit application uses the saved:

XGBoost fraud detection model

StandardScaler

The deployment code reproduces the same feature engineering and feature order used during training to keep inference consistent with the training pipeline.

Key Skills Demonstrated

Python

Pandas and NumPy

Exploratory Data Analysis

Data preprocessing

Feature engineering

Imbalanced classification

SMOTE

Scikit-learn

XGBoost

Hyperparameter tuning

Precision, Recall and F1-Score

ROC-AUC and PR-AUC

Confusion matrix analysis

Model serialization

Streamlit deployment

Author

Muhammad Muzammal Hussain

AI Engineer | Machine Learning | Deep Learning | AI Applications