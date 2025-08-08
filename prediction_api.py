# prediction_api.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_model.pkl')
print("[INFO] Model loaded for API.")

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    # Convert data into DataFrame
    df = pd.DataFrame(data, index=[0])
    # Make prediction
    prediction_proba = model.predict_proba(df)[:, 1]
    
    return jsonify({'fraud_probability': prediction_proba[0]})

if __name__ == '__main__':
    app.run(port=5000, debug=True)