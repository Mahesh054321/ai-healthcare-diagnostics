from flask import Flask, render_template, request
import numpy as np
import joblib
from generative_ai.explainer import generate_diabetes_explanation, generate_heart_explanation
from generative_ai.llm_service import generate_ai_medical_explanation
from explainable_ai.shap_explainer import explain_heart, explain_diabetes
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")


# FEATURES

diabetes_features = [
"pregnancies",
"glucose",
"bloodpressure",
"skinthickness",
"insulin",
"bmi",
"dpf",
"age"
]

heart_features = [
"age",
"sex",
"cp",
"trestbps",
"chol",
"fbs",
"restecg",
"thalach",
"exang",
"oldpeak",
"slope",
"ca",
"thal"
]

app = Flask(__name__)


# Load models with error handling
try:
    diabetes_model = joblib.load("models/diabetes_model.pkl")
    diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
    heart_model = joblib.load("models/heart_model.pkl")
    heart_scaler = joblib.load("models/heart_scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")


# Home Page
@app.route("/")
def home():
    """Render the home page with links to prediction tools."""
    return render_template("index.html")


# Diabetes Prediction

@app.route("/diabetes", methods=["GET","POST"])
def diabetes() -> str:
    """Handle diabetes prediction form and result rendering."""
    if request.method == "POST":

        features = [float(x) for x in request.form.values()]
        features = np.array([features])


        try:
            import json
            scaled = diabetes_scaler.transform(features)
            prediction = diabetes_model.predict(scaled)
            if prediction[0] == 1:
                result = "Diabetes Detected"
            else:
                result = "No Diabetes"
            risk_score = round(diabetes_model.predict_proba(scaled)[0][1]*100)
            shap_values = explain_diabetes(features[0], diabetes_features)
            patient_data = request.form.to_dict()
            explanation = generate_diabetes_explanation(result, patient_data)
            ai_text = generate_ai_medical_explanation(result, patient_data)
            shap_names = json.dumps(diabetes_features)
            # Extract the correct 1D array for SHAP values
            if hasattr(shap_values, 'values'):
                vals = shap_values.values
            else:
                vals = shap_values
            # If vals is 3D, take vals[0,:,0]; if 2D, take vals[0]; if 1D, use as is
            if hasattr(vals, 'shape'):
                if len(vals.shape) == 3:
                    vals = vals[0,:,0]
                elif len(vals.shape) == 2:
                    vals = vals[0]
            shap_vals = json.dumps([float(v) for v in vals])
            return render_template("result.html",
                prediction=result,
                explanation=explanation,
                ai_explanation=ai_text,
                risk=risk_score,
                shap_names=shap_names,
                shap_values=shap_vals
            )
        except Exception as e:
            return f"An error occurred during diabetes prediction: {e}", 500

    return render_template("diabetes.html")



# Heart Prediction

@app.route("/heart", methods=["GET","POST"])
def heart():
    """Handle heart disease prediction form and result rendering."""

    if request.method == "POST":

        data = request.form.to_dict()

        # convert text to numbers
        if data["sex"].lower() == "male":
            data["sex"] = 1
        else:
            data["sex"] = 0
        if data["exang"].lower() == "yes":
            data["exang"] = 1
        else:
            data["exang"] = 0

        # Ensure features are in the correct order and numeric
        features = []
        for feat in heart_features:
            val = data[feat]
            try:
                features.append(float(val))
            except Exception:
                features.append(0.0)
        features = np.array([features])  # 2D array


        try:
            import json
            scaled = heart_scaler.transform(features)
            prediction = heart_model.predict(scaled)
            if prediction[0] == 0:
                result = "Heart Disease Detected"
            else:
                result = "No Heart Disease"
            risk_score = round(heart_model.predict_proba(scaled)[0][0]*100)
            shap_values = explain_heart(features, heart_features)
            patient_data = request.form.to_dict()

            explanation = generate_heart_explanation(result, patient_data)
            ai_text = generate_ai_medical_explanation(result, patient_data)

# ---------- SAFE SHAP EXTRACTION ----------
            vals = shap_values.values if hasattr(shap_values, "values") else shap_values

            vals = np.array(vals)

            if vals.ndim == 3:
                vals = vals[0,:,0]
            elif vals.ndim == 2:
                vals = vals[0]
            elif vals.ndim == 1:
                vals = vals

            shap_names = json.dumps(heart_features)
            shap_vals = json.dumps([float(x) for x in vals])
            return render_template("result.html",
                prediction=result,
                explanation=explanation,
                ai_explanation=ai_text,
                risk=risk_score,
                shap_names=shap_names,
                shap_values=shap_vals
            )
        except Exception as e:
            return f"An error occurred during heart prediction: {e}", 500

    return render_template("heart.html")

if __name__ == "__main__":
    app.run(debug=True)