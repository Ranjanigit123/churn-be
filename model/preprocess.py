import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(path):
    df = pd.read_csv(path)
    df.drop(['customerID'], axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(0, inplace=True)

    label_enc = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_enc.fit_transform(df[col])

    X = df.drop('Churn', axis=1).values
    y = df['Churn'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = np.expand_dims(X, axis=2)  # for LSTM/CNN input
    return X, y
