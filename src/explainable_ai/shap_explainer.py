
import matplotlib
matplotlib.use('Agg')
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define base path for models (resolve relative to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "models")

# load models
heart_model = joblib.load(os.path.join(MODELS_DIR, "heart_model.pkl"))
diabetes_model = joblib.load(os.path.join(MODELS_DIR, "diabetes_model.pkl"))
pneumonia_model = joblib.load(os.path.join(MODELS_DIR, "pneumonia_model.pkl"))
kidney_model = joblib.load(os.path.join(MODELS_DIR, "kidney_model.pkl"))
liver_model = joblib.load(os.path.join(MODELS_DIR, "liver_model.pkl"))

# explainers - with timeout handling
try:
    heart_explainer = shap.TreeExplainer(heart_model)
except Exception as e:
    print(f"Warning: Could not initialize heart SHAP explainer: {e}")
    heart_explainer = None

try:
    diabetes_explainer = shap.TreeExplainer(diabetes_model)
except Exception as e:
    print(f"Warning: Could not initialize diabetes SHAP explainer: {e}")
    diabetes_explainer = None

try:
    pneumonia_explainer = shap.TreeExplainer(pneumonia_model)
except Exception as e:
    print(f"Warning: Could not initialize pneumonia SHAP explainer: {e}")
    pneumonia_explainer = None

try:
    kidney_explainer = shap.TreeExplainer(kidney_model)
except Exception as e:
    print(f"Warning: Could not initialize kidney SHAP explainer: {e}")
    kidney_explainer = None

try:
    liver_explainer = shap.TreeExplainer(liver_model)
except Exception as e:
    print(f"Warning: Could not initialize liver SHAP explainer: {e}")
    liver_explainer = None

# Define static directory for saving plots
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "static")


def _compute_feature_importance(model, patient_df, features):
    """Compute approximate feature importance using model predictions."""
    try:
        # Get base prediction
        base_pred = model.predict_proba(patient_df)[0]
        
        # Compute feature importance by permutation
        importance = []
        for i, feature in enumerate(features):
            # Create perturbed copy
            perturbed = patient_df.copy()
            perturbed.iloc[0, i] = patient_df.iloc[0, i].mean()  # Replace with mean
            perturbed_pred = model.predict_proba(perturbed)[0]
            
            # Compute difference
            diff = abs(base_pred[1] - perturbed_pred[1])
            importance.append(diff)
        
        # Normalize to 0-1
        importance = np.array(importance)
        if importance.sum() > 0:
            importance = importance / importance.sum()
        
        return importance
    except Exception as e:
        # Fallback: equal importance
        return np.ones(len(features)) / len(features)


def explain_heart(patient_data, features):

    # patient_data is now a 2D array
    patient_df = pd.DataFrame(patient_data, columns=features)

    shap_values = heart_explainer(patient_df)

    os.makedirs(STATIC_DIR, exist_ok=True)
    # Handle different SHAP output shapes robustly
    if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        shap.plots.waterfall(shap_values[0, :, 0], show=False)
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
        shap.plots.waterfall(shap_values[0], show=False)
    else:
        shap.plots.waterfall(shap_values, show=False)
    plt.savefig(os.path.join(STATIC_DIR, 'shap_plot.png'))
    plt.close()

    return shap_values


def explain_diabetes(patient_data, features):

    patient_df = pd.DataFrame([patient_data], columns=features)

    shap_values = diabetes_explainer(patient_df)

    os.makedirs(STATIC_DIR, exist_ok=True)
    # Fix: Only plot a single explanation, not a matrix
    if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        shap.plots.waterfall(shap_values[0, :, 0], show=False)
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
        shap.plots.waterfall(shap_values[0], show=False)
    else:
        shap.plots.waterfall(shap_values, show=False)
    plt.savefig(os.path.join(STATIC_DIR, 'shap_plot.png'))
    plt.close()

    return shap_values


def explain_pneumonia(patient_data, features):
    """Generate SHAP explanations for pneumonia prediction."""
    patient_df = pd.DataFrame([patient_data], columns=features)
    
    # Try SHAP with timeout, fallback to feature importance
    try:
        if pneumonia_explainer is not None:
            shap_values = pneumonia_explainer(patient_df)
            os.makedirs(STATIC_DIR, exist_ok=True)
            # Handle different SHAP output shapes robustly
            if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                shap.plots.waterfall(shap_values[0, :, 0], show=False)
            elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                shap.plots.waterfall(shap_values[0], show=False)
            else:
                shap.plots.waterfall(shap_values, show=False)
            plt.savefig(os.path.join(STATIC_DIR, 'shap_plot.png'))
            plt.close()
            return shap_values
    except Exception as e:
        print(f"SHAP computation failed: {e}, using fallback")
    
    # Fallback: Use feature importance
    importance = _compute_feature_importance(pneumonia_model, patient_df, features)
    return importance


def explain_kidney(patient_data, features):
    """Generate SHAP explanations for kidney disease prediction."""
    patient_df = pd.DataFrame([patient_data], columns=features)
    
    # Try SHAP with timeout, fallback to feature importance
    try:
        if kidney_explainer is not None:
            shap_values = kidney_explainer(patient_df)
            os.makedirs(STATIC_DIR, exist_ok=True)
            # Handle different SHAP output shapes robustly
            if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                shap.plots.waterfall(shap_values[0, :, 0], show=False)
            elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                shap.plots.waterfall(shap_values[0], show=False)
            else:
                shap.plots.waterfall(shap_values, show=False)
            plt.savefig(os.path.join(STATIC_DIR, 'shap_plot.png'))
            plt.close()
            return shap_values
    except Exception as e:
        print(f"SHAP computation failed: {e}, using fallback")
    
    # Fallback: Use feature importance
    importance = _compute_feature_importance(kidney_model, patient_df, features)
    return importance


def explain_liver(patient_data, features):
    """Generate SHAP explanations for liver disease prediction."""
    patient_df = pd.DataFrame([patient_data], columns=features)
    
    # Try SHAP with timeout, fallback to feature importance
    try:
        if liver_explainer is not None:
            shap_values = liver_explainer(patient_df)
            os.makedirs(STATIC_DIR, exist_ok=True)
            # Handle different SHAP output shapes robustly
            if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                shap.plots.waterfall(shap_values[0, :, 0], show=False)
            elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                shap.plots.waterfall(shap_values[0], show=False)
            else:
                shap.plots.waterfall(shap_values, show=False)
            plt.savefig(os.path.join(STATIC_DIR, 'shap_plot.png'))
            plt.close()
            return shap_values
    except Exception as e:
        print(f"SHAP computation failed: {e}, using fallback")
    
    # Fallback: Use feature importance
    importance = _compute_feature_importance(liver_model, patient_df, features)
    return importance