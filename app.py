# app.py

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# Configure the app to look for templates in the same directory
app = Flask(__name__, template_folder='.')

# --- Load Model and Scaler ---
# These must be in the same directory as this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")

try:
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")

# --- Define Feature Order ---
# This is the exact feature list from your notebook, including engineered features.
# The order must match the DataFrame columns used for training the scaler.
FEATURE_ORDER = [
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
    'cholesterol', 'gluc', 'smoke', 'alco', 'active',
    'bmi', 'pulse_pressure', 'is_hypertension'
]

# --- App Routes ---

@app.route("/")
def home():
    """Renders the main HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles prediction requests from the frontend."""
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON."}), 400

    try:
        # --- Feature Engineering (matches the notebook) ---
        # Convert age from years to days, as per the dataset's likely unit
        data['age'] = float(data.get("age", 0)) * 365.25

        # Calculate BMI
        height_m = float(data.get("height", 0)) / 100
        weight_kg = float(data.get("weight", 0))
        data['bmi'] = weight_kg / (height_m ** 2) if height_m > 0 else 0
        
        # Calculate Pulse Pressure
        ap_hi = float(data.get("ap_hi", 0))
        ap_lo = float(data.get("ap_lo", 0))
        data['pulse_pressure'] = ap_hi - ap_lo
        
        # Determine Hypertension status
        data['is_hypertension'] = 1 if (ap_hi >= 140 or ap_lo >= 90) else 0

        # Create a DataFrame with all features in the correct order
        df_input = pd.DataFrame([data])
        # Ensure the DataFrame has all columns in the correct order, filling missing ones with 0
        df_input = df_input.reindex(columns=FEATURE_ORDER, fill_value=0)
        
        # Scale the data and make a prediction
        arr_scaled = scaler.transform(df_input)
        pred_label = model.predict(arr_scaled)[0]
        pred_proba = model.predict_proba(arr_scaled)[0].tolist()

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "prediction": int(pred_label),
        "has_disease": int(pred_label),
        "probabilities": pred_proba
    })

if __name__ == "__main__":
    app.run(debug=True)