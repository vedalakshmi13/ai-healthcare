import numpy as np
import joblib

model = joblib.load("app/models/heart_model.pkl")
scaler = joblib.load("app/models/scaler.pkl")
label_encoders = joblib.load("app/models/label_encoders.pkl")

def predict_heart_disease(*input_data):
    feature_names = [
        "Disease", "Fever", "Cough", "Fatigue", "Difficulty Breathing",
        "Age", "Gender", "Blood Pressure", "Cholesterol Level"
    ]

    encoded_input = []
    for i, feature in enumerate(feature_names):
        if feature in label_encoders:
            try:
                encoded_val = label_encoders[feature].transform([input_data[i]])[0]
            except Exception as e:
                raise ValueError(f"Invalid input '{input_data[i]}' for feature '{feature}'. Error: {e}")
            encoded_input.append(encoded_val)
        else:
            encoded_input.append(float(input_data[i]))

    input_scaled = scaler.transform([encoded_input])
    prediction = model.predict(input_scaled)[0]

    result = label_encoders["Outcome Variable"].inverse_transform([prediction])[0]
    return result
