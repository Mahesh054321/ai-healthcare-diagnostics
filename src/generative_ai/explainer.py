def generate_heart_explanation(result, data):

    chol = float(data.get("chol", 0))
    bp = float(data.get("trestbps", 0))

    if "Detected" in result:
        return f"High heart risk. Cholesterol ({chol}) or BP ({bp}) may be high."
    else:
        return f"Low heart risk. Maintain healthy lifestyle."


def generate_diabetes_explanation(result, data):

    glucose = float(data.get("glucose", 0))
    bmi = float(data.get("bmi", 0))

    if "Detected" in result:
        return f"Diabetes risk present. Glucose ({glucose}) and BMI ({bmi}) are high."
    else:
        return f"Low diabetes risk. Continue healthy diet and exercise."