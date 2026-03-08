from flask import Flask, render_template, request
import numpy as np
import joblib
from generative_ai.explainer import generate_diabetes_explanation, generate_heart_explanation
from explainable_ai.shap_explainer import explain_heart, explain_diabetes


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

# Load models
diabetes_model = joblib.load("models/diabetes_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")

heart_model = joblib.load("models/heart_model.pkl")
heart_scaler = joblib.load("models/heart_scaler.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Diabetes Prediction

@app.route("/diabetes", methods=["GET","POST"])
def diabetes():
    if request.method == "POST":

        features = [float(x) for x in request.form.values()]
        features = np.array([features])

        scaled = diabetes_scaler.transform(features)

        prediction = diabetes_model.predict(scaled)

        if prediction[0] == 1:
            result = "Diabetes Detected"
        else:
            result = "No Diabetes"

        shap_values = explain_diabetes(features[0], diabetes_features)    

        patient_data = request.form.to_dict()
        explanation = generate_diabetes_explanation(result, patient_data)

        return render_template("diabetes.html", prediction=result, explanation=explanation)

    return render_template("diabetes.html")



# Heart Prediction

@app.route("/heart", methods=["GET","POST"])
def heart():

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

        features = [float(x) for x in data.values()]
        features = np.array([features])

        scaled = heart_scaler.transform(features)

        prediction = heart_model.predict(scaled)

        if prediction[0] == 0:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease"

        shap_values = explain_heart(features[0], heart_features)

        patient_data = request.form.to_dict()
        explanation = generate_heart_explanation(result, patient_data)            

        return render_template("heart.html", prediction=result, explanation=explanation)

    return render_template("heart.html")

if __name__ == "__main__":
    app.run(debug=True)