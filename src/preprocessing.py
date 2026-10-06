
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(data_path):

    # ============================================================
    # 1. Load dataset
    # ============================================================

    df = pd.read_csv(data_path)

    print("\nDataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # ============================================================
    # 2. Remove duplicate rows
    # ============================================================

    before = len(df)

    df.drop_duplicates(inplace=True)

    after = len(df)

    print("Duplicates removed:", before - after)

    # ============================================================
    # 3. Feature Engineering
    # ============================================================

    # Convert Time into hour of day
    df["Hour"] = (df["Time"] // 3600) % 24

    # Log transform transaction amount
    df["Amount_log"] = np.log1p(df["Amount"])

    # ============================================================
    # 4. Separate features and target
    # ============================================================

    X = df.drop("Class", axis=1)

    y = df["Class"]

    print("\nClass distribution:")
    print(y.value_counts())

    # ============================================================
    # 5. Train-Test Split
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ============================================================
    # 6. Feature Scaling
    # ============================================================

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train = scaler.fit_transform(X_train)

    # Transform test data using the same scaler
    X_test = scaler.transform(X_test)

    print("\nScaling completed.")

    # ============================================================
    # 7. Return processed data
    # ============================================================

    return X_train, X_test, y_train, y_test, scaler
