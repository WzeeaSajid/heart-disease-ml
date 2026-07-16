import os
import joblib
import numpy as np
import gradio as gr
import pandas as pd
from dotenv import load_dotenv

pipe = joblib.load("models/heart_pipeline.joblib")
threshold = joblib.load("models/threshold.joblib")


def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    row = pd.DataFrame(
        {
            "age": [int(age)],
            "sex": [int(sex)],
            "cp": [int(cp)],
            "trestbps": [int(trestbps)],
            "chol": [int(chol)],
            "fbs": [int(fbs)],
            "restecg": [int(restecg)],
            "thalach": [int(thalach)],
            "exang": [int(exang)],
            "oldpeak": [float(oldpeak)],
            "slope": [int(slope)],
            "ca": [int(ca)],
            "thal": [int(thal)],
        }
    )

    row["age_chol"] = row["age"] * row["chol"]
    row["thalach_age"] = row["thalach"] / row["age"]
    row["trestbps_chol"] = row["trestbps"] * row["chol"]
    row["age_thalach"] = row["age"] * row["thalach"]
    row["oldpeak_thalach"] = row["oldpeak"] * row["thalach"]
    row["age_bin"] = pd.cut(row["age"], bins=[0, 40, 55, 70, 100], labels=["0", "1", "2", "3"]).astype(str)
    row["chol_bin"] = pd.cut(row["chol"], bins=[0, 200, 240, 300, 600], labels=["0", "1", "2", "3"]).astype(str)

    proba = pipe.predict_proba(row)[0]
    disease_proba = float(proba[1])
    pred = 1 if disease_proba >= threshold else 0
    confidence = float(proba[pred])

    level = "Low"
    if confidence >= 0.75:
        level = "High"
    elif confidence >= 0.6:
        level = "Moderate"

    if pred == 1:
        accent = "#dc2626"
        bg = "#fef2f2"
        border = "#fecaca"
        icon = "&#9888;"
        label = "Heart Disease Detected"
    else:
        accent = "#16a34a"
        bg = "#f0fdf4"
        border = "#bbf7d0"
        icon = "&#10004;"
        label = "No Heart Disease"

    html = f"""
    <div style="
        background: {bg};
        border: 2px solid {border};
        border-radius: 16px;
        padding: 24px;
        font-family: system-ui, -apple-system, sans-serif;
        text-align: center;
    ">
        <div style="font-size: 48px; margin-bottom: 4px;">{icon}</div>
        <div style="
            font-size: 28px;
            font-weight: 700;
            color: {accent};
            margin-bottom: 16px;
        ">{label}</div>

        <div style="
            background: white;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 8px;
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                font-size: 14px;
                color: #6b7280;
                margin-bottom: 6px;
            ">
                <span>Confidence</span>
                <span style="font-weight: 600; color: {accent};">{confidence:.0%}</span>
            </div>
            <div style="
                background: #e5e7eb;
                border-radius: 999px;
                height: 10px;
                overflow: hidden;
            ">
                <div style="
                    width: {confidence * 100}%;
                    height: 100%;
                    background: {accent};
                    border-radius: 999px;
                    transition: width 0.4s ease;
                "></div>
            </div>
        </div>

        <div style="
            display: flex;
            justify-content: center;
            gap: 16px;
            font-size: 13px;
            color: #6b7280;
        ">
            <span>Disease Risk: <strong style="color: #dc2626;">{disease_proba:.0%}</strong></span>
            <span>&#183;</span>
            <span>Confidence: <strong style="color: {accent};">{level}</strong></span>
        </div>
    </div>
    """

    return html


with gr.Blocks(title="Heart Disease Prediction") as demo:
    gr.Markdown("# Heart Disease Prediction")
    gr.Markdown("Enter patient details to predict heart disease risk.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Patient Info")
            age = gr.Slider(20, 80, value=50, step=1, label="Age")
            sex = gr.Dropdown(choices=[("Female", 0), ("Male", 1)], label="Sex", value=1)
            cp = gr.Dropdown(
                choices=[
                    ("Typical Angina", 1),
                    ("Atypical Angina", 2),
                    ("Non-Anginal Pain", 3),
                    ("Asymptomatic", 4),
                ],
                label="Chest Pain Type",
                value=4,
            )
            trestbps = gr.Slider(80, 200, value=130, step=1, label="Resting Blood Pressure (mm Hg)")
            chol = gr.Slider(100, 600, value=240, step=1, label="Serum Cholesterol (mg/dl)")
            fbs = gr.Dropdown(choices=[("No", 0), ("Yes", 1)], label="Fasting Blood Sugar > 120", value=0)
            restecg = gr.Dropdown(
                choices=[
                    ("Normal", 0),
                    ("ST-T Wave Abnormality", 1),
                    ("LV Hypertrophy", 2),
                ],
                label="Resting ECG Results",
                value=0,
            )
        with gr.Column():
            gr.Markdown("### Exercise Test Info")
            thalach = gr.Slider(60, 220, value=150, step=1, label="Max Heart Rate Achieved")
            exang = gr.Dropdown(choices=[("No", 0), ("Yes", 1)], label="Exercise Induced Angina", value=0)
            oldpeak = gr.Slider(0.0, 6.2, value=1.0, step=0.1, label="ST Depression (oldpeak)")
            slope = gr.Dropdown(
                choices=[
                    ("Upsloping", 1),
                    ("Flat", 2),
                    ("Downsloping", 3),
                ],
                label="ST Segment Slope",
                value=2,
            )
            ca = gr.Slider(0, 3, value=0, step=1, label="# Major Vessels (0-3)")
            thal = gr.Dropdown(
                choices=[
                    ("Normal", 3),
                    ("Fixed Defect", 6),
                    ("Reversible Defect", 7),
                ],
                label="Thalassemia",
                value=3,
            )

    predict_btn = gr.Button("Predict", variant="primary")
    output = gr.HTML(label="Prediction")

    predict_btn.click(
        fn=predict,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        outputs=output,
    )

    gr.Examples(
        examples=[
            [65, 1, 4, 150, 300, 1, 1, 100, 1, 3.0, 2, 2, 7],
            [35, 0, 1, 120, 200, 0, 0, 170, 0, 0.0, 1, 0, 3],
            [55, 1, 3, 140, 250, 0, 1, 130, 0, 1.5, 2, 1, 6],
            [45, 0, 2, 130, 220, 0, 0, 160, 0, 0.0, 1, 0, 3],
        ],
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        label="Example Patients",
    )

if __name__ == "__main__":
    load_dotenv()
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
