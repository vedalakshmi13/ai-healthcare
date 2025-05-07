import numpy as np
import joblib

# ✅ Load models and encoders
model = joblib.load("app/models/heart_model.pkl")
scaler = joblib.load("app/models/scaler.pkl")
label_encoders = joblib.load("app/models/label_encoders.pkl")

# ✅ Explicitly define which features are categorical
categorical_features = set(label_encoders.keys()) - {"Outcome Variable"}

def predict_heart_disease(*input_data):
    encoded_input = []
    feature_names = list(label_encoders.keys())
    feature_names.remove("Outcome Variable")  # Remove label encoder for the target

    for i, val in enumerate(input_data):
        feature_name = feature_names[i]
        if feature_name in categorical_features:
            try:
                encoded_val = label_encoders[feature_name].transform([val])[0]
            except Exception as e:
                raise ValueError(f"Invalid input '{val}' for categorical feature '{feature_name}'. Error: {e}")
            encoded_input.append(encoded_val)
        else:
            try:
                encoded_input.append(float(val))
            except ValueError:
                raise ValueError(f"Expected a numeric value for feature '{feature_name}', got '{val}'")

    input_scaled = scaler.transform([encoded_input])
    prediction = model.predict(input_scaled)[0]

    outcome_encoder = label_encoders["Outcome Variable"]
    result = outcome_encoder.inverse_transform([prediction])[0]
    return result
