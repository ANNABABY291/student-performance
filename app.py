from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Student Performance Predictor API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

