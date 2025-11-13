# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sys

CSV = "india_supply_chain_2024_2025.csv"
if len(sys.argv) > 1:
    CSV = sys.argv[1]

df = pd.read_csv(CSV)

# Naive numeric detection
num_cols = df.select_dtypes(include=['number']).columns.tolist()
if len(num_cols) < 2:
    raise SystemExit("Need at least 2 numeric columns in the dataset to train.")

# Default: last numeric col is target
target = num_cols[-1]
features = num_cols[:-1]

X = df[features].dropna()
y = df[target].loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

joblib.dump({"model": model, "features": features, "target": target}, "exact_trained_model.pkl")
print("Saved exact_trained_model.pkl — features:", features, "target:", target)
