import gradio as gr
from app.predict import predict_heart_disease

def launch_interface():
    inputs = [
        gr.Number(label="Age"),
        gr.Radio([0, 1], label="Sex (1=Male, 0=Female)"),
        gr.Radio([0, 1, 2, 3], label="Chest Pain Type"),
        gr.Number(label="Resting Blood Pressure"),
        gr.Number(label="Cholesterol"),
        gr.Radio([0, 1], label="Fasting Blood Sugar > 120 mg/dl"),
        gr.Radio([0, 1, 2], label="Resting ECG"),
        gr.Number(label="Max Heart Rate Achieved"),
        gr.Radio([0, 1], label="Exercise Induced Angina"),
        gr.Number(label="Oldpeak (ST depression)"),
        gr.Radio([0, 1, 2], label="Slope of Peak Exercise ST Segment"),
        gr.Number(label="Number of Major Vessels Colored (0-3)"),
        gr.Radio([0, 1, 2, 3], label="Thalassemia (0=unknown, 1=normal, 2=fixed, 3=reversible)")
    ]

    gr.Interface(
        fn=predict_heart_disease,
        inputs=inputs,
        outputs="text",
        title="AI-Powered Heart Disease Prediction"
    ).launch()
