import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_lstm_model(dataframe):
    target = 'Potability'

    # Drop rows where target is missing
    data = dataframe.dropna(subset=[target])

    X = data.drop(columns=[target])
    y = data[target]

    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Reshape for LSTM [samples, timesteps, features]
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.3, random_state=42, stratify=y)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2,
              callbacks=[EarlyStopping(patience=3)], verbose=0)

    # Predict
    y_pred_probs = model.predict(X_test).ravel()
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    return {
        'Model': 'LSTM',
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4)
    }
