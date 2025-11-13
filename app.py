# ==============================
# ADVANCED Streamlit app.py
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB = True
except:
    XGB = False

st.set_page_config(page_title="Advanced Supply Chain ML App", layout="wide")
st.title("📦 Advanced Supply Chain Prediction System")

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv("india_supply_chain_2024_2025.csv")
    except:
        return None

@st.cache_data
def load_uploaded(file):
    return pd.read_csv(file)

# Sidebar - Data Loading
st.sidebar.header("1️⃣ Load Dataset")
default = st.sidebar.checkbox("Use default dataset", True)
upload = st.sidebar.file_uploader("Or upload CSV", type=["csv"])

df = None
if default:
    df = load_default_data()
if upload is not None:
    df = load_uploaded(upload)
if df is None:
    st.error("No dataset available. Upload or add default dataset.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==============================
# Target Selection
# ==============================
st.sidebar.header("2️⃣ Select Target Column")
target = st.sidebar.selectbox("Target Column", df.columns)
features = [c for c in df.columns if c != target]

# Detect task type
if df[target].dtype in ['float64','int64'] and df[target].nunique() > 20:
    task = "regression"
else:
    task = "classification"

st.sidebar.write(f"Detected Task: **{task}**")

# ==============================
# EDA
# ==============================
st.header("📊 Exploratory Data Analysis")
if st.checkbox("Show Summary"):
    st.write(df.describe(include='all').T)

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

col1, col2 = st.columns(2)
with col1:
    if len(num_cols) > 0:
        col = st.selectbox("Numeric column histogram", num_cols)
        fig = px.histogram(df, x=col, nbins=40)
        st.plotly_chart(fig)
with col2:
    if len(cat_cols) > 0:
        col = st.selectbox("Categorical column count", cat_cols)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

if st.checkbox("Correlation Heatmap") and len(num_cols) > 1:
    fig = px.imshow(df[num_cols].corr(), text_auto=True)
    st.plotly_chart(fig)

# ==============================
# Preprocessing
# ==============================
st.header("⚙️ Preprocessing Setup")
selected_features = st.multiselect("Select Features", features, default=features)

num_feats = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
cat_feats = df[selected_features].select_dtypes(exclude=[np.number]).columns.tolist()

num_imputer = "mean"
cat_imputer = "most_frequent"

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy=num_imputer)),
        ("scaler", StandardScaler())
    ]), num_feats),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy=cat_imputer)),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ]), cat_feats)
])

# ==============================
# Model Selection
# ==============================
st.header("🤖 Model Training")
model_name = st.selectbox("Choose Model", ["RandomForest", "Linear/Logistic"] + (["XGBoost"] if XGB else []))

if model_name == "RandomForest":
    model = RandomForestRegressor() if task == "regression" else RandomForestClassifier()
elif model_name == "XGBoost" and XGB:
    model = XGBRegressor() if task == "regression" else XGBClassifier()
else:
    model = LinearRegression() if task == "regression" else LogisticRegression(max_iter=1000)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# Train-test split
X = df[selected_features]
y = df[target]

if st.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    st.success("Model Trained Successfully!")

    # Metrics
    st.subheader("📌 Evaluation Metrics")
    if task == "regression":
        st.write("MSE:", mean_squared_error(y_test, pred))
        st.write("R²:", r2_score(y_test, pred))
    else:
        st.write("Accuracy:", accuracy_score(y_test, pred))
        st.write("F1 Score:", f1_score(y_test, pred, average='weighted'))

    # Save model
    joblib.dump(pipe, "model.pkl")
    with open("model.pkl", "rb") as f:
        st.download_button("Download model.pkl", f, file_name="model.pkl")

# ==============================
# Prediction Section
# ==============================
st.header("🔮 Make Predictions")

if st.checkbox("Single Prediction Input"):
    input_data = {}
    for col in selected_features:
        if col in num_feats:
            input_data[col] = st.number_input(col, value=float(df[col].median()))
        else:
            input_data[col] = st.selectbox(col, df[col].unique())

    if st.button("Predict Output"):
        model = joblib.load("model.pkl")
        df_input = pd.DataFrame([input_data])
        out = model.predict(df_input)[0]
        st.success(f"Predicted Value: {out}")

# Batch Prediction
st.subheader("📁 Batch Prediction (CSV)")
batch = st.file_uploader("Upload CSV for Prediction", type=["csv"], key="batch_csv")

if batch is not None:
    batch_df = pd.read_csv(batch)
    model = joblib.load("model.pkl")
    batch_df["prediction"] = model.predict(batch_df[selected_features])
    st.dataframe(batch_df.head())
   
