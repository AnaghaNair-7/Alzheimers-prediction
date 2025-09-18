# Alzheimer’s Disease Prediction – Flask Deployment

This project trains a **Random Forest Classifier** on Alzheimer’s disease data, saves the model with **pickle**, and deploys it using **Flask** as the backend and **HTML/CSS** as the frontend.

---

## 📂 Project Structure

ml-flask-deploy/
├─ data/
│ └─ alzheimers_disease_data.csv # Dataset file
├─ models/
│ ├─ rf_model.pkl # Trained model
│ ├─ scaler.pkl # StandardScaler
│ └─ feature_order.pkl # Feature order for input
├─ templates/
│ ├─ welcome.html # Page 1 – Welcome
│ ├─ input.html # Page 2 – Input form
│ └─ output.html # Page 3 – Prediction result
├─ static/
│ └─ styles.css # CSS styling
├─ model.py # Script to train and save model
├─ app.py # Flask backend
├─ requirements.txt # Python dependencies
└─ README.md # Project documentation

yaml
Copy code

---

## ⚙️ Setup Instructions

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/yourusername/alzheimers-flask-deploy.git
   cd alzheimers-flask-deploy
Create a Virtual Environment and Install Requirements

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Prepare the Dataset

Place your dataset file inside the data/ folder.

Ensure it is named alzheimers_disease_data.csv.

The dataset must include columns:

Copy code
PatientID, Age, Gender, Ethnicity, EducationLevel, BMI,
Smoking, AlcoholConsumption, PhysicalActivity, DietQuality,
MemoryComplaints, BehavioralProblems, ADL, Confusion, Disorientation,
PersonalityChanges, DifficultyCompletingTasks, Forgetfulness,
DoctorInCharge, Diagnosis
Train and Save the Model

bash
Copy code
python model.py
This will create rf_model.pkl, scaler.pkl, and feature_order.pkl inside the models/ folder.

Run the Flask App

bash
Copy code
python app.py
Open your browser at http://127.0.0.1:5000/

🖥️ Application Flow
Welcome Page (welcome.html)

Intro page, asks for doctor’s name, and links to input page.

Input Page (input.html)

Form where you enter patient details (Age, Gender, BMI, etc.).

Output Page (output.html)

Displays the prediction result:

Presence = Alzheimer’s disease present

Absence = Alzheimer’s disease not present

Shows confidence percentage if available.

✅ Requirements
Python 3.8+

Flask

pandas

numpy

scikit-learn

Install with:

bash
Copy code
pip install -r requirements.txt
🚀 Future Improvements
Add better categorical encoding (one-hot or label encoding).

Deploy to cloud (Heroku, AWS, or Render).

Add user authentication for doctors.

Create a database for patient record storage.

👩‍⚕️ Author
Developed for educational purposes to demonstrate ML model deployment with Flask.

yaml
Copy code

---

Would you like me to also generate the **`requirements.txt`** file code so you can run everything without dependency issues?






Ask ChatGPT




