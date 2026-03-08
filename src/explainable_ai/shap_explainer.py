import shap
import joblib
import pandas as pd

# load models
heart_model = joblib.load("models/heart_model.pkl")
diabetes_model = joblib.load("models/diabetes_model.pkl")

# explainers
heart_explainer = shap.Explainer(heart_model)
diabetes_explainer = shap.Explainer(diabetes_model)


def explain_heart(patient_data, features):

    patient_df = pd.DataFrame([patient_data], columns=features)

    shap_values = heart_explainer(patient_df)

    return shap_values


def explain_diabetes(patient_data, features):

    patient_df = pd.DataFrame([patient_data], columns=features)

    shap_values = diabetes_explainer(patient_df)

    return shap_values