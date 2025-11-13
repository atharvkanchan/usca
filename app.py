# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Supply Chain Management (Exact Clone - Clean)",
                   layout="wide", page_icon="📦")

# -------------------- Helpers --------------------
@st.cache_data
def read_csv_safe(path_or_buffer):
    """
    Attempts to read CSV robustly: standard read, fallback to latin1 & python engine.
    Coerces comma thousands and numeric-like strings into numbers where possible.
    """
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception:
        df = pd.read_csv(path_or_buffer, encoding="latin1", engine="python")
    # Try to coerce numeric-like columns
    for c in df.columns:
        if df[c].dtype == object:
            # remove commas and try convert
            try:
                cleaned = df[c].astype(str).str.replace(",", "").str.strip()
                coerced = pd.to_numeric(cleaned, errors="ignore")
                if coerced.dtype.kind in "biufc":  # numeric
                    df[c] = coerced
            except Exception:
                pass
    return df

def detect_columns(df):
    """
    Detects common supply chain columns by name variants.
    Returns mapping dict with keys: product, supplier, quantity,
    inventory, order_date, lead_time, demand, category
    """
    cols_lower = {c.lower(): c for c in df.columns}
    def find(variants):
        for v in variants:
            if v.lower() in cols_lower:
                return cols_lower[v.lower()]
        # partial matches
        for k, v in cols_lower.items():
            for opt in variants:
                if opt.lower() in k:
                    return v
        return None

    mapping = {
        "product": find(["product","item","sku","item_name","product_name","productid"]),
        "supplier": find(["supplier","vendor","vendor_name","supplier_name"]),
        "quantity": find(["quantity","qty","units","order_quantity","quantity_ordered","sold"]),
        "inventory": find(["inventory","stock","stock_level","on_hand","inventory_level"]),
        "order_date": find(["order_date","date","orderdate","transaction_date","created_at"]),
        "lead_time": find(["lead_time","leadtime","lt","lead_time_days"]),
        "demand": find(["demand","demand_qty","demand_quantity","sales","sales_quantity","demand_forecast"]),
        "category": find(["category","type","product_type","segment"])
    }
    return mapping

def show_mapping_ui(df, mapping):
    """
    Sidebar UI to let the user confirm or override detected mapping.
    Returns updated mapping.
    """
    st.sidebar.markdown("### Column mapping (confirm / override)")
    cols = ["None"] + list(df.columns)
    newmap = {}
    for key, val in mapping.items():
        default = val if val is not None else "None"
        sel = st.sidebar.selectbox(f"{key}", options=cols, index=cols.index(default) if default in cols else 0, key=f"map_{key}")
        newmap[key] = None if sel == "None" else sel
    return newmap

def safe_plot_timeseries(df, date_col, value_col, ax=None, title=None):
    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
    df2 = df2.dropna(subset=[date_col])
    if df2.empty:
        st.info("No parsable dates available for timeseries.")
        return None
    df2 = df2.sort_values(date_col)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    df2.set_index(date_col)[value_col].plot(ax=ax)
    if title:
        ax.set_title(title)
    return ax

# -------------------- Load dataset --------------------
st.sidebar.title("Data")
st.sidebar.markdown("Place file `india_supply_chain_2024_2025.csv` in this folder or use uploader.")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        df = read_csv_safe(uploaded)
        st.sidebar.success("Uploaded dataset loaded")
    except Exception as e:
        st.sidebar.error("Failed to read uploaded CSV: " + str(e))
        st.stop()
elif os.path.exists("india_supply_chain_2024_2025.csv"):
    df = read_csv_safe("india_supply_chain_2024_2025.csv")
    st.sidebar.success("Loaded india_supply_chain_2024_2025.csv")
else:
    st.sidebar.info("No dataset found. Upload or place india_supply_chain_2024_2025.csv here.")
    st.stop()

# Detect columns and let user remap
detected = detect_columns(df)
mapping = show_mapping_ui(df, detected)

# -------------------- Top header --------------------
st.title("Supply Chain Management System — Exact Clone (Clean)")
st.markdown("A reproducible supply-chain dashboard. Use the sidebar to change dataset or column mapping.")

# Sidebar navigation
page = st.sidebar.radio("Pages", ["Dashboard", "Inventory", "Suppliers", "Orders", "Forecast & ML", "Raw Data"])

# -------------------- Dashboard --------------------
if page == "Dashboard":
    st.header("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    if mapping.get("product"):
        c2.metric("Unique Products", df[mapping["product"]].nunique())
    else:
        c2.metric("Unique Products", "N/A")
    if mapping.get("inventory"):
        try:
            c3.metric("Avg Inventory", round(float(df[mapping["inventory"]].mean()), 2))
        except Exception:
            c3.metric("Avg Inventory", "N/A")
    else:
        c3.metric("Avg Inventory", "N/A")
    if mapping.get("demand"):
        try:
            c4.metric("Avg Demand", round(float(df[mapping["demand"]].mean()), 2))
        except Exception:
            c4.metric("Avg Demand", "N/A")
    else:
        c4.metric("Avg Demand", "N/A")

    st.markdown("### Top products by total quantity (if available)")
    if mapping.get("product") and mapping.get("quantity"):
        grp = df.groupby(mapping["product"])[mapping["quantity"]].sum().sort_values(ascending=False).head(15)
        st.bar_chart(grp)
    else:
        st.info("Product and/or Quantity column not detected.")

    st.markdown("### Numeric correlation (if enough numeric columns)")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(numeric.corr(), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation matrix.")

# -------------------- Inventory --------------------
elif page == "Inventory":
    st.header("Inventory Analysis")
    if not mapping.get("product") or not mapping.get("inventory"):
        st.info("Inventory view requires Product and Inventory columns. Use the mapping in the sidebar.")
    else:
        prod_list = df[mapping["product"]].dropna().unique().tolist()
        prod = st.selectbox("Select product", prod_list[:500])
        sub = df[df[mapping["product"]] == prod]
        st.write(f"Showing latest 200 rows for {prod}")
        st.dataframe(sub[[mapping["order_date"], mapping["inventory"], mapping["quantity"]] if mapping.get("order_date") in df.columns else [mapping["inventory"], mapping.get("quantity")]].drop_duplicates().head(200))

        # If date exists, show timeseries
        if mapping.get("order_date") and mapping["order_date"] in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            try:
                df_sub = sub.copy()
                df_sub[mapping["order_date"]] = pd.to_datetime(df_sub[mapping["order_date"]], errors="coerce")
                df_ts = df_sub.dropna(subset=[mapping["order_date"]]).sort_values(mapping["order_date"])
                if not df_ts.empty:
                    df_ts = df_ts.set_index(mapping["order_date"])
                    if mapping.get("inventory") and mapping["inventory"] in df_ts.columns:
                        df_ts[mapping["inventory"]].plot(ax=ax, title=f"{prod} inventory over time")
                    elif mapping.get("quantity") and mapping["quantity"] in df_ts.columns:
                        df_ts[mapping["quantity"]].plot(ax=ax, title=f"{prod} quantity over time")
                    st.pyplot(fig)
                else:
                    st.info("No valid dates for this product.")
            except Exception as e:
                st.error("Plot failed: " + str(e))

# -------------------- Suppliers --------------------
elif page == "Suppliers":
    st.header("Supplier Performance & Analysis")
    if not mapping.get("supplier"):
        st.info("Supplier column not detected. Map the supplier column in the sidebar.")
    else:
        sup_col = mapping["supplier"]
        if mapping.get("quantity"):
            agg = df.groupby(sup_col)[mapping["quantity"]].sum().sort_values(ascending=False).head(100)
            st.dataframe(agg.to_frame("total_quantity"))
            st.bar_chart(agg)
        else:
            counts = df[sup_col].value_counts().head(100)
            st.dataframe(counts.to_frame("count"))
            st.bar_chart(counts)

# -------------------- Orders --------------------
elif page == "Orders":
    st.header("Orders Timeline")
    if not mapping.get("order_date"):
        st.info("Order date not detected. Map order_date in sidebar.")
    else:
        date_col = mapping["order_date"]
        df[date_col + "_parsed"] = pd.to_datetime(df[date_col], errors="coerce")
        df_dates = df.dropna(subset=[date_col + "_parsed"])
        if df_dates.empty:
            st.info("No parsable dates found in order date column.")
        else:
            # show counts per week
            freq = st.selectbox("Resample frequency", ["D", "W", "M", "Q"], index=1)
            series = df_dates.set_index(date_col + "_parsed").resample(freq).size()
            st.line_chart(series)

            # Show recent orders table and allow filtering
            st.subheader("Recent orders")
            recent = df_dates.sort_values(date_col + "_parsed", ascending=False).head(200)
            st.dataframe(recent[[date_col, mapping.get("product"), mapping.get("supplier"), mapping.get("quantity")]].drop_duplicates().head(200))

# -------------------- Forecast & ML --------------------
elif page == "Forecast & ML":
    st.header("Forecasting & Prediction")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns to train a model. Found: " + ", ".join(numeric_cols))
    else:
        st.markdown("### Train RandomForest to predict a numeric target (Demand or similar).")
        # Try to default to demand column
        target_default = mapping.get("demand") if mapping.get("demand") in numeric_cols else numeric_cols[-1]
        target = st.selectbox("Target (y)", numeric_cols, index=numeric_cols.index(target_default) if target_default in numeric_cols else len(numeric_cols)-1)
        features = st.multiselect("Features (x)", [c for c in numeric_cols if c != target], default=[c for c in numeric_cols if c != target][:4])
        test_size = st.slider("Test size (%)", 10, 40, 20)
        train_button = st.button("Train RandomForest")

        if train_button:
            X = df[features].dropna()
            y = df[target].loc[X.index]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
            model = RandomForestRegressor(n_estimators=300, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = np.mean((preds - y_test) ** 2)
            st.session_state["exact_model"] = {"model": model, "features": features, "target": target}
            st.success(f"Model trained — Test MSE: {mse:.4f}")

        if "exact_model" in st.session_state:
            st.subheader("Make a prediction")
            info = st.session_state["exact_model"]
            model = info["model"]
            feats = info["features"]
            cols = st.columns(3)
            inp = {}
            for i, f in enumerate(feats):
                with cols[i % 3]:
                    mn = float(df[f].min()) if not pd.isna(df[f].min()) else 0.0
                    mx = float(df[f].max()) if not pd.isna(df[f].max()) else mn + 1.0
                    md = float(df[f].mean()) if not pd.isna(df[f].mean()) else mn
                    inp[f] = st.number_input(f, min_value=mn, max_value=mx, value=md)
            if st.button("Predict (Exact)"):
                import pandas as _pd
                val = model.predict(_pd.DataFrame([inp]))[0]
                st.success(f"Predicted {info['target']}: {val:.4f}")

            # Option to save model
            if st.button("Save model to exact_model.pkl"):
                joblib.dump(model, "exact_model.pkl")
                st.info("Model saved as exact_model.pkl in app folder.")

# -------------------- Raw Data --------------------
elif page == "Raw Data":
    st.header("Raw dataset (first 500 rows)")
    st.dataframe(df.head(500))
    if st.button("Download CSV (first 1000 rows)"):
        csv = df.head(1000).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="export_head1000.csv", mime="text/csv")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("Developed by Atharv • Exact clone (clean) — map columns in sidebar if detection is off.")
