import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def train_lstm_cnn_model(dataset):
    # Target column
    target = 'Potability'

    # Drop rows with missing target
    data = dataset.dropna(subset=[target])

    # Separate features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Reshape data to 3D for LSTM+CNN input
    X_reshaped = np.reshape(X_imputed, (X_imputed.shape[0], X_imputed.shape[1], 1))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.3, random_state=42, stratify=y)

    # Model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_split=0.2, callbacks=[early_stop], verbose=0)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)

    # Evaluate
    acc = accuracy_score(y_test, y_pred_class)
    prec = precision_score(y_test, y_pred_class, zero_division=0)
    rec = recall_score(y_test, y_pred_class, zero_division=0)
    f1 = f1_score(y_test, y_pred_class, zero_division=0)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    return {
        'Model': 'LSTM+CNN',
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4)
    }
