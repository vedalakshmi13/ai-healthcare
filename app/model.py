import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def train_and_save_model():
    df = pd.read_csv("data/dataset.csv")

    # Encode categorical columns
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Features and target
    X = df.drop("Outcome Variable", axis=1)
    y = df["Outcome Variable"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model, scaler, encoders
    os.makedirs("app/models", exist_ok=True)
    joblib.dump(model, "app/models/heart_model.pkl")
    joblib.dump(scaler, "app/models/scaler.pkl")
    joblib.dump(label_encoders, "app/models/label_encoders.pkl")
