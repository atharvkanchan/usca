# FINAL WORKING app.py (No Matplotlib, Only Plotly)
# 100% Compatible With Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---- Safe Plotly Import ----
import plotly.express as px

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")
st.title("📦 Supply Chain Analytics & ML Dashboard — Final Working Version")

# ----------------------------------------
# LOAD DATASET
# ----------------------------------------
st.sidebar.header("1️⃣ Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
DEFAULT_PATH = "india_supply_chain_2024_2025.csv"

# Load function
def load_df(file):
    if file is not None:
        return pd.read_csv(file)
    if os.path.exists(DEFAULT_PATH):
        return pd.read_csv(DEFAULT_PATH)
    return None

# Load dataset
df = load_df(file)
if df is None:
    st.error("❌ No dataset found. Upload a CSV or place `india_supply_chain_2024_2025.csv` in the folder.")
    st.stop()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Store trained columns for prediction\# Store trained columns for prediction
if 'trained_columns' not in st.session_state:
    st.session_state['trained_columns'] = None

# ----------------------------------------
# DASHBOARD SECTION
# ----------------------------------------
st.header("📊 Dashboard Insights")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

col1, col2 = st.columns(2)

with col1:
    if num_cols:
        Y = st.selectbox("Select Numeric Column", num_cols)
        fig1 = px.line(df, y=Y, title=f"Trend of {Y}")
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    if cat_cols:
        X = st.selectbox("Select Category Column", cat_cols)
        fig2 = px.histogram(df, x=X, title=f"Count of {X}")
        st.plotly_chart(fig2, use_container_width=True)

# Correlation heatmap
if len(num_cols) > 1:
    st.subheader("🔗 Correlation Heatmap")
    fig_corr = px.imshow(df[num_cols].corr(), text_auto=True)
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------------------------
# MACHINE LEARNING SECTION
# ----------------------------------------
st.header("🤖 ML Model Training")

target = st.selectbox("Select Target Column", df.columns)
features = [c for c in df.columns if c != target]
selected_features = st.multiselect("Select Input Features", features, default=features)

X = df[selected_features]
y = df[target]

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Detect task
if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
    task = "regression"
else:
    task = "classification"

st.info(f"Detected task type: **{task.upper()}**")

model_name = st.selectbox("Select Model", ["RandomForest", "Baseline"])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

if st.button("🚀 Train Model"):
    X_proc = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

    # Model selection
    if model_name == "RandomForest":
        model = RandomForestRegressor() if task == "regression" else RandomForestClassifier()
    else:
        class DummyModel:
            def fit(self, X, y):
                self.val = y.mean() if task == "regression" else y.mode()[0]
            def predict(self, X): return [self.val]*len(X)
        model = DummyModel()

    # Train
    model.fit(X_train, y_train)

    # Save trained model + columns
    st.session_state['trained_model'] = model
    st.session_state['trained_columns'] = X_proc.columns.tolist()

    # Evaluate
    preds = model.predict(X_test)

    st.success("Model trained successfully!")
    if task == "regression":
        st.write("MSE:", mean_squared_error(y_test, preds))
        st.write("R2 Score:", r2_score(y_test, preds))
    else:
        st.write("Accuracy:", accuracy_score(y_test, preds))
        st.write("F1 Score:", f1_score(y_test, preds, average='weighted'))

    # Save model
    joblib.dump({"model": model, "columns": st.session_state['trained_columns']}, "model.pkl")
    st.download_button("Download Trained model.pkl", open("model.pkl", "rb"), "model.pkl")

# ----------------------------------------
# SINGLE PREDICTION
# ----------------------------------------
st.header("🔮 Single Prediction")

if 'trained_model' not in st.session_state:
    st.warning("Train a model first to enable prediction.")
else:
    model = st.session_state['trained_model']
    trained_cols = st.session_state['trained_columns']

    input_data = {}
    for col in selected_features:
        if col in numeric_features:
            input_data[col] = st.number_input(col, value=float(df[col].median()))
        else:
            input_data[col] = st.selectbox(col, df[col].unique())

    if st.button("Predict Now"):
        X_input = pd.DataFrame([input_data])
        X_input = pd.get_dummies(X_input)

        # Align columns
        for c in trained_cols:
            if c not in X_input:
                X_input[c] = 0
        X_input = X_input[trained_cols]

        pred = model.predict(X_input)[0]
        st.success(f"Predicted Output: **{pred}**")

# ----------------------------------------
# BATCH PREDICTION
# ----------------------------------------
st.header("📁 Batch Prediction")
batch = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"], key="batch_csv")

if batch is not None:
    batch_df = pd.read_csv(batch)
    model = st.session_state['trained_model']
    trained_cols = st.session_state['trained_columns']

    Xb = pd.get_dummies(batch_df[selected_features])

    for c in trained_cols:
        if c not in Xb:
            Xb[c] = 0

    Xb = Xb[trained_cols]

    batch_df['prediction'] = model.predict(Xb)
    st.dataframe(batch_df.head())

    st.download_button("Download Batch Predictions", batch_df.to_csv(index=False), "batch_predictions.csv")
