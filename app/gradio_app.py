import os
import joblib

import gradio as gr
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipe      = joblib.load(os.path.join(BASE_DIR, "models", "heart_pipeline.joblib"))
threshold = joblib.load(os.path.join(BASE_DIR, "models", "threshold.joblib"))


def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    try:
        age = max(20, min(80, int(age)))
        sex = int(sex)
        cp = int(cp)
        trestbps = max(80, min(200, int(trestbps)))
        chol = max(100, min(600, int(chol)))
        fbs = int(fbs)
        restecg = int(restecg)
        thalach = max(60, min(220, int(thalach)))
        exang = int(exang)
        oldpeak = max(0.0, min(6.2, float(oldpeak)))
        slope = int(slope)
        ca = max(0, min(3, int(ca)))
        thal = int(thal)

        row = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp,
            "trestbps": trestbps, "chol": chol, "fbs": fbs,
            "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }])
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
    except Exception:
        return '<div style="text-align:center;padding:24px;color:#6B7280;font-size:13px;">Invalid input. Please check your values and try again.</div>'

    conf_level = "High" if confidence >= 0.75 else ("Moderate" if confidence >= 0.6 else "Low")

    if pred == 1:
        accent = "#DC2626"
        label = "Heart Disease Detected"
        icon = "⚠"
    else:
        accent = "#16A34A"
        label = "No Heart Disease"
        icon = "✓"

    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;max-width:420px;margin:12px auto 0;">

      <div style="text-align:center;padding:28px 20px 20px;background:#FAFBFC;
                  border:1px solid #E5E7EB;border-radius:16px;">
        <div style="font-size:36px;margin-bottom:8px;">{icon}</div>
        <div style="font-size:20px;font-weight:700;color:{accent};margin-bottom:16px;">
          {label}
        </div>
        <div style="display:inline-flex;align-items:center;gap:6px;
                    background:{accent}12;color:{accent};
                    font-size:12px;font-weight:600;padding:5px 14px;
                    border-radius:999px;">
          {conf_level} Confidence ({confidence:.0%})
        </div>
      </div>

      <div style="margin-top:12px;background:#FAFBFC;border:1px solid #E5E7EB;border-radius:16px;padding:18px 20px;">
        <div style="display:flex;justify-content:space-between;">
          <span style="font-size:13px;color:#4B5563;">Heart Disease Risk</span>
          <span style="font-size:14px;font-weight:700;color:#DC2626;">{disease_proba:.0%}</span>
        </div>
      </div>

      <div style="margin-top:10px;background:#FFF8E6;border:1px solid #FDE68A;border-radius:10px;
                  padding:10px 14px;font-size:11px;color:#92400E;line-height:1.5;">
        <strong>For informational purposes only.</strong>
        Not a substitute for professional medical advice.
      </div>
    </div>"""


CSS = """
.gradio-container { background: #F7F9FC !important; }
.gradio-container, .gradio-container * {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.gr-group, [data-testid="group"], div.block.border {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    border: 1px solid #E5E7EB !important;
    padding: 16px 18px !important;
    margin-bottom: 10px !important;
}
.block label > span {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #1F2937 !important;
}
input[type="range"] { accent-color: #DC2626 !important; }
select {
    border-radius: 8px !important;
    border: 1.5px solid #D1D5DB !important;
    font-size: 14px !important;
    padding: 8px 10px !important;
}
button.primary {
    background: #DC2626 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    color: #fff !important;
    padding: 12px 24px !important;
}
button.secondary {
    background: white !important;
    border: 1.5px solid #D1D5DB !important;
    border-radius: 10px !important;
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 20px !important;
}
footer { display: none !important; }
"""


with gr.Blocks(title="Heart Disease Risk Predictor", css=CSS) as demo:

    gr.HTML("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <div style="font-family:'Inter',system-ui,sans-serif;text-align:center;padding:28px 16px 12px;">
      <div style="font-size:32px;margin-bottom:10px;">🫀</div>
      <h1 style="font-size:22px;font-weight:700;color:#111827;margin:0 0 4px;">
        Heart Disease Risk Predictor</h1>
      <p style="color:#6B7280;font-size:13px;margin:0;line-height:1.5;">
        RF · XGBoost · LR voting ensemble
      </p>
      <p style="color:#9CA3AF;font-size:12px;margin:10px 0 0;">
        Fill in all 4 tabs below, then click Predict
      </p>
    </div>""")

    with gr.Tabs():
        with gr.Tab("Demographics"):
            age = gr.Slider(20, 80, value=50, step=1, label="Age (years)")
            sex = gr.Radio(choices=[("Female", 0), ("Male", 1)], label="Sex", value=1)

        with gr.Tab("Clinical"):
            cp = gr.Dropdown(
                choices=[("Typical Angina", 1), ("Atypical Angina", 2),
                         ("Non-Anginal Pain", 3), ("Asymptomatic", 4)],
                label="Chest Pain Type", value=4,
            )
            trestbps = gr.Slider(80, 200, value=130, step=1, label="Resting BP (mm Hg)")
            chol = gr.Slider(100, 600, value=240, step=1, label="Cholesterol (mg/dl)")
            fbs = gr.Radio(choices=[("Normal (≤ 120)", 0), ("Elevated (> 120)", 1)], label="Fasting Blood Sugar", value=0)
            restecg = gr.Dropdown(
                choices=[("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)],
                label="Resting ECG", value=0,
            )

        with gr.Tab("Exercise"):
            thalach = gr.Slider(60, 220, value=150, step=1, label="Max Heart Rate")
            exang = gr.Radio(choices=[("No", 0), ("Yes", 1)], label="Exercise Angina", value=0)
            oldpeak = gr.Number(minimum=0.0, maximum=6.2, value=1.0, step=0.1, label="ST Depression (oldpeak)")
            slope = gr.Dropdown(
                choices=[("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)],
                label="ST Slope", value=2,
            )

        with gr.Tab("Imaging"):
            ca = gr.Dropdown(
                choices=[("0", 0), ("1", 1), ("2", 2), ("3", 3)],
                label="Major Vessels (0–3)", value=0,
            )
            thal = gr.Dropdown(
                choices=[("Normal", 3), ("Fixed Defect", 6), ("Reversible Defect", 7)],
                label="Thalassemia", value=3,
            )

    with gr.Row():
        clear_btn   = gr.ClearButton(value="Clear", scale=1)
        predict_btn = gr.Button("Predict Risk", variant="primary", scale=3)

    with gr.Column():
        output = gr.HTML()

    all_inputs = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]
    predict_btn.click(fn=predict, inputs=all_inputs, outputs=output)
    clear_btn.add(all_inputs + [output])


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    port = int(os.getenv("PORT", 8080))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.red,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
