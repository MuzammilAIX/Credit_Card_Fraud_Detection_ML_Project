# Credit Card Fraud Detection

An end-to-end machine learning system for detecting fraudulent credit card transactions using a tuned XGBoost classifier, imbalanced-data handling, fraud-focused evaluation, and a Streamlit application.

## Project Highlights

* Complete workflow from data analysis to deployment
* Severe class imbalance addressed with class weighting and SMOTE experimentation
* Comparison of Logistic Regression, Decision Tree, Random Forest, and XGBoost
* XGBoost hyperparameter tuning with GridSearchCV and PR-AUC scoring
* 0.961 precision, 0.779 recall, and 0.861 F1-score on the test set
* Streamlit interface for transaction-level fraud prediction

## Results

The final model is a tuned XGBoost classifier.

| Metric    |  Score |
| --------- | -----: |
| Precision | 0.9610 |
| Recall    | 0.7789 |
| F1-Score  | 0.8605 |
| ROC-AUC   | 0.9805 |
| PR-AUC    | 0.8246 |

Confusion matrix at a 0.50 decision threshold:

```text
[[56648,     3],
 [   21,    74]]
```

The model identified 74 fraudulent transactions correctly with 3 false positives on the test set.

## Application

The Streamlit application provides:

* Fraud probability
* Fraudulent or legitimate classification
* Model information
* Prediction threshold

## Dataset

The project uses the standard Credit Card Fraud Detection dataset containing:

* `Time` — seconds elapsed between transactions
* `V1` to `V28` — PCA-transformed transaction features
* `Amount` — transaction amount
* `Class` — target variable

  * `0` = Legitimate
  * `1` = Fraudulent

The dataset is highly imbalanced, so precision, recall, F1-score, and PR-AUC are emphasized over accuracy alone.

## Machine Learning Workflow

```text
Data Understanding
        ↓
Data Quality Checks
        ↓
Duplicate Removal
        ↓
Exploratory Data Analysis
        ↓
Class Imbalance Analysis
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
Model Comparison
        ↓
XGBoost
        ↓
Hyperparameter Tuning
        ↓
Evaluation
        ↓
Model Serialization
        ↓
Streamlit Deployment
```

### Feature Engineering

* `Hour` extracted from transaction `Time`
* `Amount_log` created using `log1p(Amount)`

### Final XGBoost Configuration

```text
n_estimators = 200
max_depth = 7
learning_rate = 0.1
subsample = 1.0
colsample_bytree = 0.8
```

Hyperparameter tuning used `GridSearchCV` with `average_precision` as the scoring metric.

## Project Structure

```text
Credit_Card_Fraud_Detection_ML_Project/
│
├── app/
│   └── app.py
├── data/
│   └── raw/
├── images/
│   └── screenshots/
├── models/
├── notebooks/
│   └── credit_card_analysis.ipynb
├── reports/
├── src/
│   ├── __init__.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Tech Stack

**Languages and Libraries**

Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, imbalanced-learn, XGBoost, Joblib

**Deployment**

Streamlit

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/MuzammilAIX/Credit_Card_Fraud_Detection_ML_Project.git
cd Credit_Card_Fraud_Detection_ML_Project
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Train the model

```powershell
python src/train.py
```

### 5. Run the Streamlit application

```powershell
python -m streamlit run app/app.py
```

## Key Machine Learning Concepts

* Binary classification
* Highly imbalanced datasets
* Stratified train-test splitting
* Standardization
* Feature engineering
* Class weighting
* SMOTE
* Model comparison
* XGBoost
* Hyperparameter tuning
* Precision, recall, and F1-score
* ROC-AUC and PR-AUC
* Confusion matrix analysis
* Decision thresholding
* Feature importance
* Model serialization
* Streamlit deployment

## Limitations

The `V1` to `V28` features are PCA-transformed variables, so their individual business meaning is not directly interpretable.

The deployed decision threshold is `0.50`. A production fraud detection system should select the threshold according to the financial cost of false positives and false negatives.

## Author

**Muhammad Muzammal Hussain**

AI Engineer | Machine Learning | Deep Learning | AI Applications

GitHub: [MuzammilAIX](https://github.com/MuzammilAIX)
