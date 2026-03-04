from flask import Flask, render_template, request
import numpy as np
import joblib

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

        return render_template("diabetes.html", prediction=result)

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

        if prediction[0] == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease"

        return render_template("heart.html", prediction=result)

    return render_template("heart.html")

if __name__ == "__main__":
    app.run(debug=True)