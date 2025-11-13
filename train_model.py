# train_model.py
"""
Standalone training script.

Usage:
    python train_model.py [path_to_csv]

Default CSV path is 'india_supply_chain_2024_2025.csv' in current folder.

Outputs:
    - exact_trained_model.pkl  (contains dict {'model', 'features', 'target'})
"""
import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

CSV = "india_supply_chain_2024_2025.csv"
if len(sys.argv) > 1:
    CSV = sys.argv[1]

if not os.path.exists(CSV):
    raise SystemExit(f"CSV file not found: {CSV}")

print("Loading dataset:", CSV)
df = pd.read_csv(CSV)

# Coerce numeric-like columns
for c in df.columns:
    if df[c].dtype == object:
        try:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="ignore")
        except Exception:
            pass

num_cols = df.select_dtypes(include=['number']).columns.tolist()
if len(num_cols) < 2:
    raise SystemExit("Need at least 2 numeric columns to train a model. Found: " + ", ".join(num_cols))

# Default: last numeric column is target
target = num_cols[-1]
features = num_cols[:-1]

print("Detected numeric columns:", num_cols)
print("Using features:", features)
print("Using target:", target)

# Drop rows with missing values for features & target
X = df[features].dropna()
y = df[target].loc[X.index]

if len(X) < 10:
    raise SystemExit("Not enough rows after dropping NA to train. Need at least 10 valid rows.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = float(((preds - y_test) ** 2).mean())
print(f"Training complete. Test MSE: {mse:.4f}")

out = {
    "model": model,
    "features": features,
    "target": target
}
joblib.dump(out, "exact_trained_model.pkl")
print("Saved model to exact_trained_model.pkl")
