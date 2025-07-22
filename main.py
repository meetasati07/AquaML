import os
import matplotlib.pyplot as plt
from PyEMD import EMD
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.impute import SimpleImputer
from lstm import train_lstm_model
from lstm_cnn import train_lstm_cnn_model

@st.cache_data
def load_data(file_path):
    try:
        dataset = pd.read_csv(file_path)
        return dataset
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the path.")
        st.stop()


def emd_per_sample():
    # Initialize session state variable if not exists
    if "show_dataset" not in st.session_state:
        st.session_state.show_dataset = False

    dst = load_data('dataset/water_potability.csv')

    feature_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                       'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    emd_features_list = []

    for idx, row in dst[feature_columns].iterrows():
        signal = row.values.astype(float)

        emd_obj = EMD()
        imfs = emd_obj(signal)

        # Compute mean & std per IMF
        sample_features = {}
        for i, imf in enumerate(imfs):
            sample_features[f'imf_{i + 1}_mean'] = np.mean(imf)
            sample_features[f'imf_{i + 1}_std'] = np.std(imf)

        emd_features_list.append(sample_features)

    # Convert to dataframe
    emd_features_df = pd.DataFrame(emd_features_list)

    # Combine with original dataset
    dst_with_emd = pd.concat([dst.reset_index(drop=True), emd_features_df.reset_index(drop=True)], axis=1)

    button_text = "Hide Processed Dataset" if st.session_state.show_dataset else "View Processed Dataset"
    if st.button(button_text):
        st.session_state.show_dataset = not st.session_state.show_dataset

    if st.session_state.show_dataset:
        st.write("Processed Dataset:")
        st.dataframe(dst_with_emd)

    # Call train_and_evaluate_models here, after dst_with_emd is created
    metrics_df = train_and_evaluate_models(dst_with_emd)

    return dst_with_emd


def train_and_evaluate_models(dst_with_emd):
    st.subheader("🧪 Train & Compare ML Models")

    target = 'Potability'

    # Remove rows with missing target
    data = dst_with_emd.dropna(subset=[target])

    # Separate features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.3, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'SVM': SVC(probability=True)
    }

    metrics_list = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5

        metrics_list.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4)
        })

    lstm_metrics = train_lstm_model(dst_with_emd)
    metrics_list.append(lstm_metrics)

    lstm_cnn_metrics = train_lstm_cnn_model(dst_with_emd)
    metrics_list.append(lstm_cnn_metrics)

    # OPTIONAL: Simulated deep models if you still want them
    # metrics_list.append({...})  ← remove these if you're using real models now

    metrics_df = pd.DataFrame(metrics_list)

    st.write("📊 **Model Comparison Table:**")
    st.dataframe(metrics_df)

    st.subheader("📈 Accuracy Comparison")
    st.bar_chart(metrics_df.set_index('Model')['Accuracy'])

    st.subheader("🏆 Top Performer")
    best_model = metrics_df.sort_values(by='Accuracy', ascending=False).iloc[0]
    st.success(f"✅ Best model is **{best_model['Model']}** with Accuracy = {best_model['Accuracy']}")

    return metrics_df

if __name__ == "__main__":
    emd_per_sample()