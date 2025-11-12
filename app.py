#app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os

# FastAPI app initialization
app = FastAPI(title="Customer Churn Prediction API")

# Load model
MODEL_PATH = "model/churn_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}. Please run train_model.py first.")

model = tf.keras.models.load_model(MODEL_PATH)

# Dummy scaler (you can replace with joblib.load("scaler.pkl") if saved separately)
#scaler = StandardScaler()
scaler = joblib.load("model/scaler.pkl")
le_contract = joblib.load("model/le_contract.pkl")
le_internet = joblib.load("model/le_internet.pkl")
le_payment = joblib.load("model/le_payment.pkl")




# Define request body
class CustomerData(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    contract_type: str
    internet_service: str
    payment_method: str

# Encode categorical values (simple static encodings for now)
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
internet_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
payment_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer': 2, 'Credit card': 3}

@app.get("/")
def home():
    return {"message": "Welcome to the Customer Churn Prediction API 🚀"}

@app.post("/predict")
def predict(data: CustomerData):
    # Convert input to numeric form
    tenure = data.tenure
    monthly_charges = data.monthly_charges
    total_charges = data.total_charges
    contract_type = le_contract.transform([data.contract_type])[0]
    internet_service = le_internet.transform([data.internet_service])[0]
    payment_method = le_payment.transform([data.payment_method])[0]


    features = np.array([[tenure, monthly_charges, total_charges,
                          contract_type, internet_service, payment_method]])

    # Scale using pre-fitted scaler
    features_scaled = scaler.transform(features)
    features_scaled = np.expand_dims(features_scaled, axis=2)  # for Conv1D/LSTM input

    # Predict
    prediction = model.predict(features_scaled)[0][0]
    churn_prob = float(prediction)
    churn_label = "Churn" if churn_prob >= 0.5 else "No Churn"

    return {
        "churn_probability": round(churn_prob, 3),
        "prediction": churn_label
    }
