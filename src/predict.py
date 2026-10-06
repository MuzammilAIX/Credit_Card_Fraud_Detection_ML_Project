import numpy as np
import pandas as pd
import joblib


# Load trained model, scaler, and threshold
model = joblib.load("../models/xgb_fraud_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
threshold = joblib.load("../models/threshold.pkl")


def predict_fraud(transaction):
    """
    Predict whether a credit card transaction is
    legitimate (0) or fraudulent (1).
    """

    # Convert input into DataFrame
    df = pd.DataFrame([transaction])

    # Feature Engineering
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Amount_log"] = np.log1p(df["Amount"])

    # Keep the same feature order used during training
    feature_names = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Hour", "Amount_log"]
    )

    X = df[feature_names]

    # Apply the same scaler used during training
    X_scaled = scaler.transform(X)

    # Get fraud probability
    fraud_probability = model.predict_proba(X_scaled)[:, 1][0]

    # Apply final threshold = 0.5
    prediction = int(fraud_probability >= threshold)

    # Return result
    if prediction == 1:
        result = "Fraudulent Transaction"
    else:
        result = "Legitimate Transaction"

    return {
        "prediction": prediction,
        "fraud_probability": fraud_probability,
        "result": result
    }