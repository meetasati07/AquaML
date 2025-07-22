# lstm_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error


def train_lstm_model(data):
    """
    Train and evaluate an LSTM model on the given dataset with EMD features.
    Assumes 'Potability' is the target column.
    """

    target = 'Potability'

    # Drop rows with missing target
    data = data.dropna(subset=[target])

    # Separate features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Normalize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Reshape input for LSTM: (samples, timesteps, features)
    # We'll treat each sample as a sequence of 1 timestep with n features
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.3, stratify=y, random_state=42
    )

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0  # Set to 1 to see training output
    )

    # Predict and evaluate
    y_pred = model.predict(X_test).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary, zero_division=0)
    rec = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
    mse = mean_squared_error(y_test, y_pred_binary)
    rmse = mse ** 0.5

    metrics = {
        'Model': 'LSTM',
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4)
    }

    return metrics
