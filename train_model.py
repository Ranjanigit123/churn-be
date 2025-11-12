#train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, Attention, concatenate
from tensorflow.keras.optimizers import Adam
import joblib
import os

# Load dataset
data_path = "data/Telco-Customer-Churn.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

df = pd.read_csv(data_path)

# Clean and preprocess
df = df[df["TotalCharges"] != " "]
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Select features
features = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'Contract', 'InternetService', 'PaymentMethod'
]

X = df[features]
y = df["Churn"]

# Encode categorical features
le_contract = LabelEncoder()
le_internet = LabelEncoder()
le_payment = LabelEncoder()

X["Contract"] = le_contract.fit_transform(X["Contract"])
X["InternetService"] = le_internet.fit_transform(X["InternetService"])
X["PaymentMethod"] = le_payment.fit_transform(X["PaymentMethod"])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save encoders & scaler for API use
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le_contract, "model/le_contract.pkl")
joblib.dump(le_internet, "model/le_internet.pkl")
joblib.dump(le_payment, "model/le_payment.pkl")

# Reshape for CNN/LSTM input
X_scaled = np.expand_dims(X_scaled, axis=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define hybrid deep learning model
input_layer = Input(shape=(X_train.shape[1], 1))

# CNN branch
cnn = Conv1D(64, kernel_size=2, activation='relu')(input_layer)
cnn = GlobalMaxPooling1D()(cnn)

# BiLSTM + Attention branch
lstm = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
attn = Attention()([lstm, lstm])
attn = GlobalMaxPooling1D()(attn)

# Merge
merged = concatenate([cnn, attn])
dense = Dense(64, activation='relu')(merged)
dense = Dropout(0.3)(dense)
output = Dense(1, activation='sigmoid')(dense)

# Build and compile model
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
print("🚀 Training model on real Telco dataset...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("model/churn_model.h5")

print("✅ Training complete. Model saved at model/churn_model.h5")
