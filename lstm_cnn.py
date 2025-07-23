def train_lstm_cnn_model(dataframe):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    target = 'Potability'
    data = dataframe.dropna(subset=[target])
    X = data.drop(columns=[target])
    y = data[target]

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Reshape for Conv1D + LSTM
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.3, random_state=42, stratify=y)

    # Model: Deeper CNN + LSTM
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2,
              callbacks=[EarlyStopping(patience=3)], verbose=0)

    y_pred_probs = model.predict(X_test).ravel()
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
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
