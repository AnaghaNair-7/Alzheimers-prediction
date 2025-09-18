# app.py
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)
MODEL_DIR = 'models'

# Load model, scaler, feature order
with open(os.path.join(MODEL_DIR, 'rf_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'feature_order.pkl'), 'rb') as f:
    feature_order = pickle.load(f)

# Example small mapping for binary -> label
LABEL_MAP = {0: 'Presence', 1: 'Absence'}

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input', methods=['GET', 'POST'])
def input_page():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    # Render input.html and pass feature names so template can build fields dynamically
    return render_template('input.html', features=feature_order)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('input_page'))

    # Build input array in feature_order
    data = request.form
    vals = []
    for feat in feature_order:
        raw = data.get(feat)
        if raw is None or raw == '':
            val = 0.0
        else:
            try:
                val = float(raw)
            except ValueError:
                # Handle simple categorical encodings
                s = raw.strip().lower()
                if s in ['male', 'm']:
                    val = 1.0
                elif s in ['female', 'f']:
                    val = 0.0
                else:
                    # crude deterministic numeric encoding
                    val = float(abs(hash(s)) % 100)
        vals.append(val)

    X = np.array(vals).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = None
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_scaled).max()

    label = LABEL_MAP.get(int(pred), str(pred))

    return render_template('output.html', prediction=label, probability=prob)

if __name__ == '__main__':
    app.run(debug=True)
