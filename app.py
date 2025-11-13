# app.py - Streamlit Supply Chain Dashboard + Prediction System
"""
Improved & safer Streamlit dashboard + ML app.
Key fixes in this version:
- Safe imports for optional packages (plotly, joblib). If plotly is missing the app falls back to matplotlib.
- Avoid crashes when model isn't trained or model.pkl missing.
- Preserve one-hot encoded columns used during training to align single-row & batch predictions.
- Clearer error messages for missing dependencies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import warnings
warnings.filterwarnings('ignore')

# Safe import for plotly
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

# Safe import for joblib
try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    joblib = None
    JOBLIB_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")
st.title("📦 Supply Chain Analytics & ML Dashboard")

# ----------------------------------------
# LOAD DATASET
# ----------------------------------------
st.sidebar.header("1) Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
DEFAULT_PATH = "india_supply_chain_2024_2025.csv"

def load_dataframe(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if os.path.exists(DEFAULT_PATH):
        return pd.read_csv(DEFAULT_PATH)
    return None

df = load_dataframe(uploaded_file)
if df is None:
    st.error("Dataset not found. Upload CSV or place india_supply_chain_2024_2025.csv in the working folder.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Maintain global variable to store columns after one-hot during training
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
        metric_col = st.selectbox("Select Metric to Analyze", num_cols, key='metric_col')
        if PLOTLY_AVAILABLE:
            fig1 = px.line(df, y=metric_col, title=f"Trend of {metric_col}")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.line_chart(df[metric_col])
with col2:
    if cat_cols:
        cat_col = st.selectbox("Select Category to Count", cat_cols, key='cat_col')
        if PLOTLY_AVAILABLE:
            fig2 = px.histogram(df, x=cat_col, title=f"Distribution of {cat_col}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write(df[cat_col].value_counts())

st.subheader("Correlation Heatmap")
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    if PLOTLY_AVAILABLE:
        corr_fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.write(corr)
else:
    st.info("Not enough numeric columns for correlation map.")

# ----------------------------------------
# MACHINE LEARNING SECTION
# ----------------------------------------
st.header("🤖 Prediction Model")

# Target selection
target = st.selectbox("Select Target Column", df.columns.tolist(), key='target')
features = [c for c in df.columns if c != target]

st.write("Features used for training (editable):")
selected_features = st.multiselect("Select features", options=features, default=features, key='selected_features')

# Simple feature type lists
X = df[selected_features]
y = df[target]

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Detect task
if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
    task = "regression"
else:
    task = "classification"

st.write(f"Detected task: **{task}**")

model_choice = st.selectbox("Model Type", options=["RandomForest", "Baseline"], index=0, key='model_choice')

# ----------------------------------------
# TRAIN MODEL
# ----------------------------------------
if st.button("Train Model"):
    st.info("Training started — this may take a while for large datasets.")

    # One-hot encode categorical features and align columns
    X_proc = pd.get_dummies(X, drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

    if model_choice == "RandomForest":
        model = RandomForestRegressor(n_jobs=-1, random_state=42) if task == "regression" else RandomForestClassifier(n_jobs=-1, random_state=42)
    else:
        # Baseline: mean predictor for regression, most frequent for classification
        class DummyModel:
            def fit(self, X, y):
                self.is_reg = pd.api.types.is_numeric_dtype(y)
                if self.is_reg:
                    self.val = float(y.mean())
                else:
                    self.val = y.mode().iloc[0]
                return self
            def predict(self, X):
                return np.array([self.val]*len(X))
        model = DummyModel()

    model.fit(X_train, y_train)

    # Save trained columns to session for prediction alignment
    st.session_state['trained_columns'] = X_proc.columns.tolist()
    st.session_state['trained_model'] = model

    # Evaluate
    preds = model.predict(X_test)
    st.success("Model trained successfully!")
    if task == "regression":
        st.write("MSE:", mean_squared_error(y_test, preds))
        st.write("R2 Score:", r2_score(y_test, preds))
    else:
        st.write("Accuracy:", accuracy_score(y_test, preds))
        st.write("F1 Score:", f1_score(y_test, preds, average='weighted'))

    # Save model to disk if joblib available
    if JOBLIB_AVAILABLE:
        joblib.dump({'model': model, 'columns': st.session_state['trained_columns']}, 'model.pkl')
        st.success("model.pkl saved to disk.")
        with open('model.pkl', 'rb') as f:
            st.download_button("Download model.pkl", data=f, file_name="model.pkl")
    else:
        st.warning("joblib not available. Install joblib and redeploy to enable model save/load.")

# ----------------------------------------
# SINGLE PREDICTION
# ----------------------------------------
st.header("🔮 Predict with Your Model")

model_for_pred = st.session_state.get('trained_model', None)
trained_cols = st.session_state.get('trained_columns', None)

if model_for_pred is None:
    # Try to load from disk
    if JOBLIB_AVAILABLE and os.path.exists('model.pkl'):
        try:
            loaded = joblib.load('model.pkl')
            model_for_pred = loaded.get('model') if isinstance(loaded, dict) else loaded
            trained_cols = loaded.get('columns') if isinstance(loaded, dict) else None
            st.session_state['trained_model'] = model_for_pred
            st.session_state['trained_columns'] = trained_cols
            st.success('Loaded model.pkl from disk for prediction')
        except Exception as e:
            st.warning('No trained model in session or disk. Train a model to enable predictions.')

if model_for_pred is not None:
    st.subheader("Single-row prediction")
    input_data = {}
    for col in selected_features:
        if col in numeric_features:
            input_data[col] = st.number_input(col, value=float(df[col].median()), key=f"in_{col}")
        else:
            opts = df[col].dropna().unique().tolist()
            input_data[col] = st.selectbox(col, opts, key=f"in_{col}")

    if st.button("Predict"):
        try:
            X_input = pd.DataFrame([input_data])
            X_input_proc = pd.get_dummies(X_input, drop_first=False)
            # Align columns
            if trained_cols is not None:
                for c in trained_cols:
                    if c not in X_input_proc.columns:
                        X_input_proc[c] = 0
                X_input_proc = X_input_proc[trained_cols]
            pred = model_for_pred.predict(X_input_proc)
            st.success(f"Prediction: {pred[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("No model available for prediction. Train or upload a model.")

# ----------------------------------------
# BATCH PREDICTION
# ----------------------------------------
st.header("📁 Batch Prediction (CSV)")
batch_file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="batch")
if batch_file is not None:
    if model_for_pred is None and not (JOBLIB_AVAILABLE and os.path.exists('model.pkl')):
        st.error('No trained model available. Train or upload model.pkl')
    else:
        batch_df = pd.read_csv(batch_file)
        try:
            X_batch = batch_df[selected_features]
            X_batch_proc = pd.get_dummies(X_batch, drop_first=False)
            if trained_cols is not None:
                for c in trained_cols:
                    if c not in X_batch_proc.columns:
                        X_batch_proc[c] = 0
                X_batch_proc = X_batch_proc[trained_cols]
            preds = model_for_pred.predict(X_batch_proc)
            batch_df['prediction'] = preds
            st.dataframe(batch_df.head())
            csv_bytes = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", data=csv_bytes, file_name='predictions.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Batch prediction error: {e}")

st.markdown("---")
st.write("If anything else needs fixing, tell me the exact error message and I will update the file.")
