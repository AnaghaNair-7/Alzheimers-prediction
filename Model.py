# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# --- CONFIG ---
DATA_PATH = "alzheimers_disease_data.csv"   # put your dataset in the same folder
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Features (drop unnecessary columns)
X = df.drop(["Diagnosis", "PatientID", "DoctorInCharge"], axis=1, errors="ignore")
y = df["Diagnosis"]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=2
)

# Standardization
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train_scaled, y_train)

# Accuracy check
acc = model.score(x_test_scaled, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Save model, scaler, and feature order
with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

feature_order = list(X.columns)
with open(os.path.join(MODEL_DIR, "feature_order.pkl"), "wb") as f:
    pickle.dump(feature_order, f)

print("Model, scaler, and feature order saved in 'models/' folder.")
