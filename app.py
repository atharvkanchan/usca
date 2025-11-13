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

# Store trained columns for prediction\if 'trained_columns' not in st.session_state:
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
# MACHINE
