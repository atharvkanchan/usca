# Supply Chain Management 

## Overview
This is the Exact Clean Clone version of a supply-chain analytics dashboard.
It auto-detects common columns (Product, Supplier, Quantity, Inventory, Order Date, Demand) and lets you remap them from the sidebar.

## Files
- `app.py` — main Streamlit app (Exact Clean Clone full code)
- `train_model.py` — CLI script to train a RandomForest model from your CSV
- `requirements.txt` — Python dependencies
- `india_supply_chain_2024_2025.csv` — place your CSV here (or upload in UI)
- `pages/` — optional modular pages (Overview, Visual Analytics, ML Prediction, Forecasting)

## Quick start (local)
1. Create virtual environment (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux / macOS
   .venv\\Scripts\\activate     # Windows
