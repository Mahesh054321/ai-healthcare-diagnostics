from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import json
from generative_ai.explainer import generate_diabetes_explanation, generate_heart_explanation
from generative_ai.llm_service import generate_ai_medical_explanation
from explainable_ai.shap_explainer import explain_heart, explain_diabetes, explain_pneumonia, explain_kidney, explain_liver
from dotenv import load_dotenv
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

load_dotenv(dotenv_path=".env")


# FEATURES

diabetes_features = [
"Pregnancies",
"Glucose",
"BloodPressure",
"SkinThickness",
"Insulin",
"BMI",
"DiabetesPedigreeFunction",
"Age"
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

pneumonia_features = [
"age",
"sex", 
"fever",
"cough",
"breath",
"wbc",
"crp",
"oxygen"
]

kidney_features = [
"age",
"bp",
"creatinine",
"bun",
"proteinuria",
"glucose",
"albumin",
"hemoglobin"
]

liver_features = [
"age",
"bilirubin",
"albumin",
"alt",
"ast",
"alk",
"pt",
"platelets"
]

app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"),
            static_url_path="/static",
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))

# Define base path for models (resolve relative to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

# Load models with error handling
try:
    diabetes_model = joblib.load(os.path.join(MODELS_DIR, "diabetes_model.pkl"))
    diabetes_scaler = joblib.load(os.path.join(MODELS_DIR, "diabetes_scaler.pkl"))
    heart_model = joblib.load(os.path.join(MODELS_DIR, "heart_model.pkl"))
    heart_scaler = joblib.load(os.path.join(MODELS_DIR, "heart_scaler.pkl"))
    pneumonia_model = joblib.load(os.path.join(MODELS_DIR, "pneumonia_model.pkl"))
    pneumonia_scaler = joblib.load(os.path.join(MODELS_DIR, "pneumonia_scaler.pkl"))
    kidney_model = joblib.load(os.path.join(MODELS_DIR, "kidney_model.pkl"))
    kidney_scaler = joblib.load(os.path.join(MODELS_DIR, "kidney_scaler.pkl"))
    liver_model = joblib.load(os.path.join(MODELS_DIR, "liver_model.pkl"))
    liver_scaler = joblib.load(os.path.join(MODELS_DIR, "liver_scaler.pkl"))
    print(f"✓ All models loaded successfully from {MODELS_DIR}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
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

        features = [float(request.form.get(f, 0)) for f in diabetes_features]
        features_df = pd.DataFrame([features], columns=diabetes_features)


        try:
            import json
            scaled = diabetes_scaler.transform(features_df)
            prediction = diabetes_model.predict(scaled)
            if prediction[0] == 1:
                result = "Diabetes Detected"
            else:
                result = "No Diabetes"
            risk_score = round(diabetes_model.predict_proba(scaled)[0][1]*100)
            shap_values = explain_diabetes(features, diabetes_features)
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
        features_df = pd.DataFrame([features], columns=heart_features)


        try:
            import json
            scaled = heart_scaler.transform(features_df)
            prediction = heart_model.predict(scaled)
            if prediction[0] == 0:
                result = "Heart Disease Detected"
            else:
                result = "No Heart Disease"
            risk_score = round(heart_model.predict_proba(scaled)[0][0]*100)
            shap_values = explain_heart([features], heart_features)
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


# Pneumonia Prediction
@app.route("/pneumonia", methods=["GET","POST"])
def pneumonia():
    """Handle pneumonia prediction form and result rendering."""
    if request.method == "POST":
        try:
            features = [float(request.form.get(f, 0)) for f in pneumonia_features]
            features_df = pd.DataFrame([features], columns=pneumonia_features)
            
            scaled = pneumonia_scaler.transform(features_df)
            prediction = pneumonia_model.predict(scaled)
            
            if prediction[0] == 1:
                result = "Pneumonia Detected"
            else:
                result = "No Pneumonia"
            
            risk_score = round(pneumonia_model.predict_proba(scaled)[0][1] * 100)
            
            # Get SHAP/feature importance values
            shap_values = explain_pneumonia(features, pneumonia_features)
            
            # Handle both SHAP values and fallback importance arrays
            if hasattr(shap_values, 'values'):
                vals = shap_values.values
            else:
                vals = shap_values
            
            # Extract the correct 1D array
            if hasattr(vals, 'shape'):
                if len(vals.shape) == 3:
                    vals = vals[0,:,0]
                elif len(vals.shape) == 2:
                    vals = vals[0]
            
            shap_vals = json.dumps([float(v) for v in vals])
            
            patient_data = request.form.to_dict()
            explanation = f"Based on the provided symptoms and test results, the analysis indicates a {risk_score}% risk of pneumonia."
            ai_text = f"AI Analysis: The patient's symptoms suggest {'high' if risk_score > 70 else 'moderate' if risk_score > 40 else 'low'} likelihood of respiratory infection."
            shap_names = json.dumps(pneumonia_features)
            
            return render_template("result.html",
                prediction=result,
                explanation=explanation,
                ai_explanation=ai_text,
                risk=risk_score,
                shap_names=shap_names,
                shap_values=shap_vals
            )
        except Exception as e:
            import traceback
            print(f"❌ Pneumonia error: {e}")
            print(traceback.format_exc())
            return f"An error occurred during pneumonia prediction: {e}", 500

    return render_template("pneumonia.html")


# Kidney Disease Prediction
@app.route("/kidney", methods=["GET","POST"])
def kidney():
    """Handle kidney disease prediction form and result rendering."""
    if request.method == "POST":
        try:
            features = [float(request.form.get(f, 0)) for f in kidney_features]
            features_df = pd.DataFrame([features], columns=kidney_features)
            
            scaled = kidney_scaler.transform(features_df)
            prediction = kidney_model.predict(scaled)
            
            if prediction[0] == 1:
                result = "Kidney Disease Detected"
            else:
                result = "No Kidney Disease"
            
            risk_score = round(kidney_model.predict_proba(scaled)[0][1] * 100)
            
            # Get SHAP/feature importance values
            shap_values = explain_kidney(features, kidney_features)
            
            # Handle both SHAP values and fallback importance arrays
            if hasattr(shap_values, 'values'):
                vals = shap_values.values
            else:
                vals = shap_values
            
            # Extract the correct 1D array
            if hasattr(vals, 'shape'):
                if len(vals.shape) == 3:
                    vals = vals[0,:,0]
                elif len(vals.shape) == 2:
                    vals = vals[0]
            
            shap_vals = json.dumps([float(v) for v in vals])
            
            patient_data = request.form.to_dict()
            explanation = f"Based on the provided lab results and clinical data, the analysis indicates a {risk_score}% risk of kidney disease."
            ai_text = f"AI Analysis: The patient's renal function markers suggest {'high' if risk_score > 70 else 'moderate' if risk_score > 40 else 'low'} risk of chronic kidney disease."
            shap_names = json.dumps(kidney_features)
            
            return render_template("result.html",
                prediction=result,
                explanation=explanation,
                ai_explanation=ai_text,
                risk=risk_score,
                shap_names=shap_names,
                shap_values=shap_vals
            )
        except Exception as e:
            import traceback
            print(f"❌ Kidney error: {e}")
            print(traceback.format_exc())
            return f"An error occurred during kidney prediction: {e}", 500

    return render_template("kidney.html")


# Liver Disease Prediction
@app.route("/liver", methods=["GET","POST"])
def liver():
    """Handle liver disease prediction form and result rendering."""
    if request.method == "POST":
        try:
            features = [float(request.form.get(f, 0)) for f in liver_features]
            features_df = pd.DataFrame([features], columns=liver_features)
            
            scaled = liver_scaler.transform(features_df)
            prediction = liver_model.predict(scaled)
            
            if prediction[0] == 1:
                result = "Liver Disease Detected"
            else:
                result = "No Liver Disease"
            
            risk_score = round(liver_model.predict_proba(scaled)[0][1] * 100)
            
            # Get SHAP/feature importance values
            shap_values = explain_liver(features, liver_features)
            
            # Handle both SHAP values and fallback importance arrays
            if hasattr(shap_values, 'values'):
                vals = shap_values.values
            else:
                vals = shap_values
            
            # Extract the correct 1D array
            if hasattr(vals, 'shape'):
                if len(vals.shape) == 3:
                    vals = vals[0,:,0]
                elif len(vals.shape) == 2:
                    vals = vals[0]
            
            shap_vals = json.dumps([float(v) for v in vals])
            
            patient_data = request.form.to_dict()
            explanation = f"Based on the provided liver function tests and clinical data, the analysis indicates a {risk_score}% risk of liver disease."
            ai_text = f"AI Analysis: The patient's hepatic markers suggest {'high' if risk_score > 70 else 'moderate' if risk_score > 40 else 'low'} risk of liver dysfunction."
            shap_names = json.dumps(liver_features)
            
            return render_template("result.html",
                prediction=result,
                explanation=explanation,
                ai_explanation=ai_text,
                risk=risk_score,
                shap_names=shap_names,
                shap_values=shap_vals
            )
        except Exception as e:
            import traceback
            print(f"❌ Liver error: {e}")
            print(traceback.format_exc())
            return f"An error occurred during liver prediction: {e}", 500

    return render_template("liver.html")


# Symptom Checker
@app.route("/symptom-checker", methods=["GET","POST"])
def symptom_checker():
    """Handle symptom checker form and result rendering."""
    if request.method == "POST":
        symptoms = request.form.getlist("symptoms")
        # Mock analysis based on symptoms
        risk_scores = {
            "diabetes": 0,
            "heart": 0,
            "pneumonia": 0,
            "kidney": 0,
            "liver": 0
        }
        
        # Simple rule-based scoring
        if "Fever" in symptoms and "Cough" in symptoms:
            risk_scores["pneumonia"] += 30
        if "Chest Pain" in symptoms and "Shortness of Breath" in symptoms:
            risk_scores["heart"] += 40
        if "Frequent Urination" in symptoms and "Fatigue" in symptoms:
            risk_scores["diabetes"] += 25
        if "Abdominal Pain" in symptoms and "Jaundice" in symptoms:
            risk_scores["liver"] += 35
        if "Swelling" in symptoms and "Fatigue" in symptoms:
            risk_scores["kidney"] += 30
        
        # Find highest risk condition
        max_risk = max(risk_scores.values())
        if max_risk > 0:
            condition = max(risk_scores, key=risk_scores.get)
            result = f"Possible {condition.title()} Condition"
        else:
            result = "No significant condition detected"
            max_risk = 10
        
        explanation = f"Based on your reported symptoms ({', '.join(symptoms)}), the most likely condition is {result.lower()} with {max_risk}% confidence."
        ai_text = f"AI Symptom Analysis: Your symptoms suggest consulting a healthcare professional for proper diagnosis. This is not a substitute for medical advice."
        
        return render_template("result.html",
            prediction=result,
            explanation=explanation,
            ai_explanation=ai_text,
            risk=max_risk,
            shap_names='["Symptom Match","Severity","Pattern Recognition"]',
            shap_values='[40,30,30]'
        )
    return render_template("symptom_checker.html")


if __name__ == "__main__":
    app.run(debug=True)
