
import matplotlib
matplotlib.use('Agg')
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# load models
heart_model = joblib.load("models/heart_model.pkl")
diabetes_model = joblib.load("models/diabetes_model.pkl")

# explainers
heart_explainer = shap.TreeExplainer(heart_model)
diabetes_explainer = shap.TreeExplainer(diabetes_model)


def explain_heart(patient_data, features):

    # patient_data is now a 2D array
    patient_df = pd.DataFrame(patient_data, columns=features)

    shap_values = heart_explainer(patient_df)

    os.makedirs('static', exist_ok=True)
    # Handle different SHAP output shapes robustly
    if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        shap.plots.waterfall(shap_values[0, :, 0], show=False)
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
        shap.plots.waterfall(shap_values[0], show=False)
    else:
        shap.plots.waterfall(shap_values, show=False)
    plt.savefig('static/shap_plot.png')
    plt.close()

    return shap_values


def explain_diabetes(patient_data, features):

    patient_df = pd.DataFrame([patient_data], columns=features)

    shap_values = diabetes_explainer(patient_df)

    os.makedirs('static', exist_ok=True)
    # Fix: Only plot a single explanation, not a matrix
    if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        shap.plots.waterfall(shap_values[0, :, 0], show=False)
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
        shap.plots.waterfall(shap_values[0], show=False)
    else:
        shap.plots.waterfall(shap_values, show=False)
    plt.savefig('static/shap_plot.png')
    plt.close()

    return shap_values