import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

def generate_ai_medical_explanation(prediction, patient_data):
    # Simple rule-based explanations based on prediction
    pred = prediction.lower()
    if "heart disease" in pred or "heart" in pred:
        explanation = """
        **Heart Disease Risk Detected**
        Risk Level: High
        Diet Tips: Reduce salt, eat more fruits/vegetables, avoid saturated fats.
        Exercise Tips: 30 minutes of moderate exercise daily, like walking.
        Precautions: Monitor blood pressure, avoid smoking, regular check-ups.
        """
    elif "diabetes" in pred or "diabetic" in pred:
        explanation = """
        **Diabetes Risk Detected**
        Risk Level: Moderate
        Diet Tips: Low-carb diet, control sugar intake, eat balanced meals.
        Exercise Tips: Daily walks, yoga, maintain healthy weight.
        Precautions: Monitor blood sugar, consult doctor, avoid stress.
        """
    else:
        explanation = """
        **General Health Advice**
        Risk Level: Low
        Diet Tips: Balanced diet with proteins, carbs, fats.
        Exercise Tips: Regular physical activity.
        Precautions: Regular health screenings.
        """

    return explanation