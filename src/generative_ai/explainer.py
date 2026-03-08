def generate_diabetes_explanation(result, data):

    glucose = data["glucose"]
    bmi = data["bmi"]
    age = data["age"]

    if "Diabetes" in result:

        explanation = (
            f"The AI model detected possible diabetes risk. "
            f"The glucose level ({glucose}) and BMI ({bmi}) may indicate higher blood sugar levels. "
            f"Age ({age}) can also influence diabetes risk."
        )

    else:

        explanation = (
            f"The AI model did not detect strong diabetes indicators. "
            f"The glucose level ({glucose}) and BMI ({bmi}) appear within a safer range."
        )

    return explanation



def generate_heart_explanation(result, data):

    chol = data["chol"]
    bp = data["trestbps"]
    age = data["age"]

    if "Heart Disease" in result:

        explanation = (
            f"The AI model detected possible heart disease risk. "
            f"Cholesterol level ({chol}) and blood pressure ({bp}) may increase cardiovascular risk."
        )

    else:

        explanation = (
            f"The AI model predicts low heart disease risk. "
            f"Cholesterol ({chol}) and blood pressure ({bp}) appear within a normal range."
        )

    return explanation