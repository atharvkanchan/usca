# app.py - Streamlit Supply Chain Dashboard + Prediction System
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")
st.title("📦 Supply Chain Analytics & ML Dashboard")

# ----------------------------------------
# LOAD DATASET
# ----------------------------------------
st.sidebar.header("1) Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
default_path = "india_supply_chain_2024_2025.csv"

try:
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(default_path)
except:
    st.error("Dataset not found. Upload CSV or place india_supply_chain_2024_2025.csv in folder.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----------------------------------------
# DASHBOARD SECTION
# ----------------------------------------
st.header("📊 Dashboard Insights")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

col1, col2 = st.columns(2)

with col1:
    if num_cols:
        metric_col = st.selectbox("Select Metric to Analyze", num_cols)
        fig1 = px.line(df, y=metric_col, title=f"Trend of {metric_col}")
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    if cat_cols:
        cat_col = st.selectbox("Select Category to Count", cat_cols)
        fig2 = px.histogram(df, x=cat_col, title=f"Distribution of {cat_col}")
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("Correlation Heatmap")
if len(num_cols) > 1:
    corr_fig = px.imshow(df[num_cols].corr(), text_auto=True, title="Correlation Matrix")
    st.plotly_chart(corr_fig, use_container_width=True)
else:
    st.info("Not enough numeric columns for correlation map.")

# ----------------------------------------
# MACHINE LEARNING SECTION
# ----------------------------------------
st.header("🤖 Prediction Model")

target = st.selectbox("Select Target Column", df.columns)
features = [c for c in df.columns if c != target]

st.write("Features used for training:", features)

X = df[features]
y = df[target]

numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(exclude=[np.number]).columns

# Identify task
if y.dtype in ['float64', 'int64'] and y.nunique() > 20:
    task = "regression"
else:
    task = "classification"

model_type = st.selectbox("Model Type", ["RandomForest", "Linear/Logistic"])

# ----------------------------------------
# TRAIN MODEL
# ----------------------------------------
if st.button("Train Model"):
    X_processed = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    if task == "regression":
        model = RandomForestRegressor()
    else:
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.success("Model trained successfully!")

    if task == "regression":
        st.write("MSE:", mean_squared_error(y_test, preds))
        st.write("R2 Score:", r2_score(y_test, preds))
    else:
        st.write("Accuracy:", accuracy_score(y_test, preds))
        st.write("F1 Score:", f1_score(y_test, preds, average='weighted'))

    joblib.dump(model, "model.pkl")
    st.success("model.pkl saved!")
    with open("model.pkl", "rb") as f:
        st.download_button("Download model.pkl", data=f, file_name="model.pkl")

# ----------------------------------------
# SINGLE PREDICTION
# ----------------------------------------
st.header("🔮 Predict with Your Model")

if st.checkbox("Enter values for single prediction"):
    input_data = {}
    for col in features:
        if col in numeric_features:
            input_data[col] = st.number_input(col, value=float(df[col].median()))
        else:
            input_data[col] = st.selectbox(col, df[col].unique())

    if st.button("Predict"):
        try:
            model = joblib.load("model.pkl")
            X_input = pd.DataFrame([input_data])
            X_input = pd.get_dummies(X_input)
            missing_cols = set(X_processed.columns) - set(X_input.columns)
            for c in missing_cols:
                X_input[c] = 0
            X_input = X_input[X_processed.columns]
            pred = model.predict(X_input)[0]
            st.success(f"Prediction: {pred}")
        except:
            st.error("Train the model first or upload a valid model.pkl")

# ----------------------------------------
# BATCH PREDICTION
# ----------------------------------------
st.header("📁 Batch Prediction (CSV)")
batch_file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="batch")

if batch_file is not None:
    batch_df = pd.read_csv(batch_file)
    try:
        model = joblib.load("model.pkl")
        batch_proc = pd.get_dummies(batch_df)
        missing_cols = set(X_processed.columns) - set(batch_proc.columns)
        for c in missing_cols:
            batch_proc[c] = 0
        batch_proc = batch_proc[X_processed.columns]
        batch_df['prediction'] = model.predict(batch_proc)
        st.dataframe(batch_df.head())
        st.download_button("Download Predictions", batch_df.to_csv(index=False), file_name="batch_predictions.csv")
    except Exception as e:
        st.error(f"Error: {e}")
