import gradio as gr
from app.predict import predict_heart_disease

def launch_interface():
    inputs = [
        gr.Textbox(label="Disease"),
        gr.Radio(["Yes", "No"], label="Fever"),
        gr.Radio(["Yes", "No"], label="Cough"),
        gr.Radio(["Yes", "No"], label="Fatigue"),
        gr.Radio(["Yes", "No"], label="Difficulty Breathing"),
        gr.Number(label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["Low", "Normal", "High"], label="Blood Pressure"),
        gr.Radio(["Low", "Normal", "High"], label="Cholesterol Level")
    ]

    gr.Interface(
        fn=predict_heart_disease,
        inputs=inputs,
        outputs="text",
        title="AI Disease Outcome Predictor",
        description="Predict the outcome (Positive/Negative) based on symptoms and patient data."
    ).launch()
