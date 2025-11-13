# app.py
"""
Advanced Streamlit App for Supply Chain Prediction
Features:
- Upload or load default CSV
- Automatic problem type detection (regression/classification)
- Interactive EDA (summary, histograms, correlation, scatter)
- Preprocessing pipeline with imputing, scaling, encoding
- Multiple model options (RandomForest, XGBoost if available, Linear/Logistic)
- Hyperparameter tuning (GridSearchCV optional)
- Cross-validation metrics and test-set evaluation
- Feature importance (tree-based) and permutation importance
- Single-row prediction UI and batch prediction for uploaded files
- Save & download trained model (joblib)

Usage:
- Place a default dataset named `india_supply_chain_2024_2025.csv` in the working directory OR upload your CSV in the app.
- Select the target column and proceed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost, else skip
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

st.set_page_config(page_title="Advanced Supply Chain ML App", layout="wide")
st.title("📊 Advanced Supply Chain Prediction App")

# -------------------- Utilities --------------------
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv("india_supply_chain_2024_2025.csv")
    except Exception:
        return None

@st.cache_data
def read_uploaded(file) -> pd.DataFrame:
    return pd.read_csv(file)

def detect_problem_type(y: pd.Series) -> str:
    if y.dtype.kind in 'biufc' and y.nunique() > 20:
        return 'regression'
    else:
        # if numerical but small unique values, treat as classification
        if y.dtype.kind in 'biufc' and y.nunique() <= 20:
            return 'classification'
        # categorical
        return 'classification'

# -------------------- Sidebar: Data load --------------------
st.sidebar.header("1) Data")
use_default = st.sidebar.checkbox("Use default dataset (india_supply_chain_2024_2025.csv)", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type=["csv"] )

if use_default:
    df = load_default_data()
    if df is None and uploaded_file is None:
        st.sidebar.error("Default CSV not found. Please upload a CSV.")
else:
    df = None

if uploaded_file is not None:
    df = read_uploaded(uploaded_file)

if df is None:
    st.warning("No data available. Upload a CSV or place india_supply_chain_2024_2025.csv in the app folder.")
    st.stop()

st.subheader("Dataset — preview")
st.dataframe(df.head())

# -------------------- Sidebar: Target selection --------------------
st.sidebar.header("2) Target & Task")
cols = df.columns.tolist()
target = st.sidebar.selectbox("Select target column", options=cols)
problem_hint = detect_problem_type(df[target])
problem_type = st.sidebar.selectbox("Detected task type", options=["regression", "classification"], index=0 if problem_hint=='regression' else 1)
if st.sidebar.button("Re-detect task"):
    problem_type = detect_problem_type(df[target])

# -------------------- EDA --------------------
st.header("Exploratory Data Analysis (EDA)")
if st.checkbox("Show dataset info and summary"):
    st.subheader("Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.subheader("Summary statistics")
    st.write(df.describe(include='all').T)

col1, col2 = st.columns(2)
with col1:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    chosen_num = st.selectbox("Numeric column for histogram", options=numeric_cols)
    fig = px.histogram(df, x=chosen_num, nbins=30, title=f"Distribution: {chosen_num}")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    if len(cat_cols)>0:
        chosen_cat = st.selectbox("Categorical column for counts", options=cat_cols)
        fig2 = px.histogram(df, x=chosen_cat, title=f"Counts: {chosen_cat}")
        st.plotly_chart(fig2, use_container_width=True)

if st.checkbox("Show correlation heatmap"):
    corr = df[numeric_cols].corr()
    fig3 = px.imshow(corr, text_auto='.2f', title='Correlation matrix')
    st.plotly_chart(fig3, use_container_width=True)

# -------------------- Preprocessing --------------------
st.header("Preprocessing & Feature Selection")
st.write("Choose preprocessing options and features to include in the model.")
all_features = [c for c in df.columns if c != target]
selected_features = st.multiselect("Features to use", options=all_features, default=all_features)

# Determine column types from selected features
num_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
cat_features = df[selected_features].select_dtypes(include=['object','category','bool']).columns.tolist()

st.write(f"Numeric features: {num_features}")
st.write(f"Categorical features: {cat_features}")

# Imputer and encoder choices
num_impute_strategy = st.selectbox("Numeric imputer", options=["mean","median","constant"], index=0)
cat_impute_strategy = st.selectbox("Categorical imputer", options=["most_frequent","constant"], index=0)
cat_encoding = st.selectbox("Categorical encoding", options=["onehot","ordinal"], index=0)
scale_numeric = st.checkbox("Scale numeric features (StandardScaler)", value=True)

# Build transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy=num_impute_strategy)),
    ('scaler', StandardScaler() if scale_numeric else 'passthrough')
])

if cat_encoding == 'onehot':
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cat_impute_strategy, fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
else:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cat_impute_strategy, fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
], remainder='drop')

# -------------------- Model selection --------------------
st.header("Modeling")
model_choice = st.selectbox("Choose a model", options=["RandomForest","Linear/Logistic"] + (["XGBoost"] if XGBOOST_AVAILABLE else []))

params = {}
if model_choice == 'RandomForest':
    if problem_type=='regression':
        model = RandomForestRegressor(random_state=42)
        params = {
            'model__n_estimators': [100, 300],
            'model__max_depth': [None, 10, 20]
        }
    else:
        model = RandomForestClassifier(random_state=42)
        params = {
            'model__n_estimators': [100, 300],
            'model__max_depth': [None, 10, 20]
        }

elif model_choice == 'XGBoost' and XGBOOST_AVAILABLE:
    if problem_type == 'regression':
        model = XGBRegressor(random_state=42, eval_metric='rmse', use_label_encoder=False)
        params = {
            'model__n_estimators': [100, 300],
            'model__max_depth': [3,6]
        }
    else:
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        params = {
            'model__n_estimators': [100, 300],
            'model__max_depth': [3,6]
        }
else:
    if problem_type == 'regression':
        model = LinearRegression()
        params = {}
    else:
        model = LogisticRegression(max_iter=1000)
        params = {'model__C': [0.1, 1.0, 10.0]}

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Train/test split
st.sidebar.header("3) Training")
test_size = st.sidebar.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", value=42)

if st.button("Train model"):
    X = df[selected_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

    # Optionally run GridSearch
    use_grid = st.checkbox("Run GridSearchCV (may be slow)")
    if use_grid and params:
        st.write("Running GridSearchCV... this may take time")
        grid = GridSearchCV(pipeline, param_grid=params, cv=3, n_jobs=-1, scoring='r2' if problem_type=='regression' else 'accuracy')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        st.write("Best params:", grid.best_params_)
    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    # Save model in session state
    st.session_state['model'] = best_model

    # Evaluate
    y_pred = best_model.predict(X_test)
    st.subheader("Evaluation on test set")
    if problem_type == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"MSE: {mse:.4f}")
        st.write(f"R2: {r2:.4f}")
    else:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall: {rec:.4f}")
        st.write(f"F1-score: {f1:.4f}")

    # Cross-validation
    if st.checkbox("Show cross-validation scores", value=False):
        scoring = 'r2' if problem_type=='regression' else 'accuracy'
        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring=scoring)
        st.write(f"CV ({scoring}) scores: {cv_scores}")
        st.write(f"Mean CV score: {cv_scores.mean():.4f}")

    # Feature importance for tree models
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        st.subheader("Feature importances (model)")
        # Need to get feature names after preprocessing
        try:
            # get feature names from preprocessor
            ohe_cols = []
            if cat_features and cat_encoding=='onehot':
                ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                cat_names = ohe.get_feature_names_out(cat_features)
            elif cat_features and cat_encoding=='ordinal':
                cat_names = cat_features
            else:
                cat_names = []
            feature_names = list(num_features) + list(cat_names)
            importances = best_model.named_steps['model'].feature_importances_
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
            st.dataframe(fi_df.head(30))
            fig_fi = px.bar(fi_df.head(20), x='importance', y='feature', orientation='h')
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.write("Could not compute feature importances:", e)

    # Permutation importance
    if st.checkbox("Compute permutation importance (slow)"):
        st.write("Computing permutation importance on test set...")
        r = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        perm_df = pd.DataFrame({'feature': np.array(num_features+cat_features), 'importance_mean': r.importances_mean, 'importance_std': r.importances_std}).sort_values('importance_mean', ascending=False)
        st.dataframe(perm_df.head(30))

    # Save model button
    buf = io.BytesIO()
    joblib.dump(best_model, 'model.pkl')
    st.success("Model trained and saved to model.pkl")
    with open('model.pkl', 'rb') as f:
        st.download_button('Download trained model (model.pkl)', data=f, file_name='model.pkl')

# -------------------- Load existing model --------------------
st.header("Use an existing model")
uploaded_model = st.file_uploader("Upload a trained joblib model (.pkl)", type=['pkl'])
if uploaded_model is not None:
    try:
        loaded_model = joblib.load(uploaded_model)
        st.session_state['model'] = loaded_model
        st.success("Model loaded into session")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# -------------------- Prediction UI --------------------
st.header("Make Predictions")
if 'model' not in st.session_state:
    st.info('No trained model in session. Train a model or upload one to make predictions.')
else:
    model_for_pred = st.session_state['model']
    st.subheader("Single-row prediction")
    with st.form("single_pred"):
        input_dict = {}
        for feat in selected_features:
            if feat in num_features:
                input_dict[feat] = st.number_input(f"{feat}", value=float(df[feat].median()))
            else:
                opts = df[feat].dropna().unique().tolist()
                if len(opts) > 50:
                    val = st.text_input(f"{feat} (enter value)")
                else:
                    val = st.selectbox(feat, opts)
                input_dict[feat] = val
        submitted = st.form_submit_button("Predict")
        if submitted:
            X_single = pd.DataFrame([input_dict])
            try:
                pred_single = model_for_pred.predict(X_single)
                st.write("Prediction:", pred_single[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.subheader("Batch prediction for CSV")n
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'], key='batch')
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        required_cols_missing = [c for c in selected_features if c not in batch_df.columns]
        if len(required_cols_missing) > 0:
            st.error(f"Uploaded CSV missing required columns: {required_cols_missing}")
        else:
            try:
                preds = model_for_pred.predict(batch_df[selected_features])
                batch_df['prediction'] = preds
                st.dataframe(batch_df.head())
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions CSV', data=csv, file_name='predictions.csv', mime='text/csv')
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# -------------------- Footer / Help --------------------
st.markdown("---")
st.markdown("**Tips:** Use RandomForest or XGBoost for non-linear relationships, and Linear/Logistic for quick baselines. Use GridSearch only when you have time and resources.")
st.markdown("If you'd like, I can also provide a separate `train_model.py` script that trains and exports `model.pkl` using the same preprocessing pipeline.")
# =====================
"""
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
"""


