import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("C:/Users/ANAGHA/OneDrive/Desktop/AnaghaAI.Project/alzheimers_disease_data.csv")

# Feature selection
x = df.drop(["Diagnosis", "PatientID", "DoctorInCharge"], axis=1)
y = df['Diagnosis']

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)

# Standardize the features
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Initialize and train Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Save the trained model and scaler using pickle
with open("alzheimers_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")
