import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Supply Chain Management - Exact Clean Clone",
                   layout="wide", page_icon="📦")

# ------------------ SAFE CSV READER ------------------
@st.cache_data
def read_csv_safe(path_or_buffer):
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception:
        df = pd.read_csv(path_or_buffer, encoding="latin1", engine="python")
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="ignore")
            except:
                pass
    return df

# ------------------ COLUMN DETECTION ------------------
def detect_columns(df):
    cols = {c.lower(): c for c in df.columns}

    def match(keys):
        for k in keys:
            if k.lower() in cols:
                return cols[k.lower()]
        for cl in cols:
            for k in keys:
                if k.lower() in cl:
                    return cols[cl]
        return None

    return {
        "product": match(["product", "item", "sku", "product_name"]),
        "supplier": match(["supplier", "vendor", "supplier_name"]),
        "quantity": match(["quantity", "qty", "units", "order_quantity"]),
        "inventory": match(["inventory", "stock", "stock_level", "inventory_level"]),
        "order_date": match(["order_date", "date", "transaction_date"]),
        "lead_time": match(["lead_time", "lt"]),
        "demand": match(["demand", "sales", "demand_qty", "sold"]),
        "category": match(["category", "type", "product_type"])
    }

# ------------------ COLUMN MAPPING UI ------------------
def mapping_ui(df, mapping):
    st.sidebar.subheader("Column Mapping")
    new_map = {}
    columns = ["None"] + list(df.columns)

    for key, val in mapping.items():
        sel = st.sidebar.selectbox(key, options=columns,
                                   index=columns.index(val) if val in columns else 0)
        new_map[key] = None if sel == "None" else sel
    return new_map

# ------------------ LOAD DATA ------------------
st.sidebar.title("Load Dataset")
upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if upload:
    df = read_csv_safe(upload)
    st.sidebar.success("Uploaded dataset loaded.")
elif os.path.exists("india_supply_chain_2024_2025.csv"):
    df = read_csv_safe("india_supply_chain_2024_2025.csv")
    st.sidebar.success("Loaded india_supply_chain_2024_2025.csv")
else:
    st.error("Please upload a dataset or place india_supply_chain_2024_2025.csv in the folder.")
    st.stop()

# Map columns
detected = detect_columns(df)
mapping = mapping_ui(df, detected)

# ------------------ NAVIGATION ------------------
st.title("📦 Supply Chain Management Dashboard – Exact Clean Clone")
page = st.sidebar.radio("Navigation", 
                        ["Dashboard", "Inventory", "Suppliers", "Orders", "Forecast & ML", "Raw Data"])


# ------------------ PAGE: DASHBOARD ------------------
if page == "Dashboard":
    st.header("📊 Overall Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))

    if mapping["product"]:
        c2.metric("Unique Products", df[mapping["product"]].nunique())
    else:
        c2.metric("Unique Products", "N/A")

    if mapping["inventory"]:
        c3.metric("Avg Inventory", round(df[mapping["inventory"]].mean(), 2))
    else:
        c3.metric("Avg Inventory", "N/A")

    if mapping["demand"]:
        c4.metric("Avg Demand", round(df[mapping["demand"]].mean(), 2))
    else:
        c4.metric("Avg Demand", "N/A")

    st.subheader("Top Products by Quantity")
    if mapping["product"] and mapping["quantity"]:
        agg = df.groupby(mapping["product"])[mapping["quantity"]].sum().sort_values(ascending=False).head(15)
        st.bar_chart(agg)
    else:
        st.info("Product or Quantity column missing.")

    st.subheader("Correlation Heatmap (Numeric Columns)")
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(9,5))
        sns.heatmap(num.corr(), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns.")


# ------------------ PAGE: INVENTORY ------------------
elif page == "Inventory":
    st.header("📦 Inventory Analysis")

    if not mapping["product"] or not mapping["inventory"]:
        st.warning("Product and Inventory columns are required.")
    else:
        products = df[mapping["product"]].unique().tolist()
        prod = st.selectbox("Select Product", products)
        subset = df[df[mapping["product"]] == prod]

        st.write("Recent Inventory Records")
        st.dataframe(subset[[mapping["inventory"], mapping["quantity"]]].head(100))

        if mapping["order_date"]:
            st.subheader("Inventory Over Time")
            subset[mapping["order_date"]] = pd.to_datetime(subset[mapping["order_date"]], errors="coerce")
            ts = subset.dropna(subset=[mapping["order_date"]]).sort_values(mapping["order_date"])

            if not ts.empty:
                st.line_chart(ts.set_index(mapping["order_date"])[mapping["inventory"]])


# ------------------ PAGE: SUPPLIERS ------------------
elif page == "Suppliers":
    st.header("🏭 Supplier Dashboard")

    if not mapping["supplier"]:
        st.warning("Supplier column missing.")
    else:
        if mapping["quantity"]:
            sup = df.groupby(mapping["supplier"])[mapping["quantity"]].sum().sort_values(ascending=False)
            st.dataframe(sup.to_frame("Total Quantity"))
            st.bar_chart(sup.head(20))
        else:
            counts = df[mapping["supplier"]].value_counts()
            st.dataframe(counts.to_frame("Count"))
            st.bar_chart(counts.head(20))


# ------------------ PAGE: ORDERS ------------------
elif page == "Orders":
    st.header("📅 Order Timeline")

    if not mapping["order_date"]:
        st.warning("Order Date column missing.")
    else:
        df["parsed_date"] = pd.to_datetime(df[mapping["order_date"]], errors="coerce")
        clean = df.dropna(subset=["parsed_date"])

        freq = st.selectbox("Frequency", ["D", "W", "M"], index=1)
        timeline = clean.set_index("parsed_date").resample(freq).size()
        st.line_chart(timeline)

        st.subheader("Latest 200 Orders")
        st.dataframe(clean[[mapping["order_date"], mapping["product"], mapping["supplier"], mapping["quantity"]]].head(200))


# ------------------ PAGE: FORECAST & ML ------------------
elif page == "Forecast & ML":
    st.header("🤖 Forecast & Machine Learning")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("At least 2 numeric columns required.")
    else:
        target = st.selectbox("Target Column", num_cols)
        features = st.multiselect("Feature Columns", [c for c in num_cols if c != target],
                                  default=[c for c in num_cols if c != target][:4])

        if st.button("Train Model"):
            X = df[features].dropna()
            y = df[target].loc[X.index]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestRegressor(n_estimators=300, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = np.mean((preds - y_test) ** 2)

            st.session_state["model"] = model
            st.success(f"Model Trained Successfully! MSE: {mse:.4f}")

        if "model" in st.session_state:
            st.subheader("Prediction")

            model = st.session_state["model"]
            user_input = {}
            cols = st.columns(3)

            for i, f in enumerate(features):
                with cols[i % 3]:
                    user_input[f] = st.number_input(
                        f,
                        float(df[f].min()),
                        float(df[f].max()),
                        float(df[f].mean())
                    )

            if st.button("Predict"):
                import pandas as pd
                val = model.predict(pd.DataFrame([user_input]))[0]
                st.success(f"Predicted {target}: {val:.3f}")

            if st.button("Save Model"):
                joblib.dump(model, "exact_model.pkl")
                st.success("Model saved as exact_model.pkl")


# ------------------ PAGE: RAW DATA ------------------
elif page == "Raw Data":
    st.header("📄 Raw Dataset")
    st.dataframe(df.head(500))

    if st.button("Download First 1000 Rows"):
        st.download_button(
            "Download CSV",
            df.head(1000).to_csv(index=False).encode(),
            file_name="export_1000_rows.csv",
            mime="text/csv"
        )
