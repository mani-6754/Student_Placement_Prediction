import joblib
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


# Load Random Forest model
model = joblib.load("placement_random_forest.pkl")

print("Model loaded successfully!")


print("Expected features:")
print(model.feature_names_in_)

print("Number of features:")
print(model.n_features_in_)


# Home route
@app.route('/')
def home():
    return "Placement Prediction API is running!"


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    student = pd.DataFrame([{

        'IQ': float(data['IQ']),

        'Prev_Sem_Result': float(
            data['Prev_Sem_Result']
        ),

        'CGPA': float(
            data['CGPA']
        ),

        'Academic_Performance': float(
            data['Academic_Performance']
        ),

        'Internship_Experience': int(
            data['Internship_Experience']
        ),

        'Extra_Curricular_Score': float(
            data['Extra_Curricular_Score']
        ),

        'Communication_Skills': float(
            data['Communication_Skills']
        ),

        'Projects_Completed': int(
            data['Projects_Completed']
        )

    }])


    # Prediction
    prediction = model.predict(student)[0]


    # Convert 0/1 into text
    if prediction == 1:
        result = "Placed"
    else:
        result = "Not Placed"


    return jsonify({
        "prediction": result
    })


# Start Flask server
if __name__ == '__main__':

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True
    )

