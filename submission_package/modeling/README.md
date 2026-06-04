Model training and evaluation

Run the model training pipeline that trains classification and (if possible) Cox survival models.

Prerequisites
- Create a Python environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\Activate on Windows
pip install -r modeling/requirements.txt
```

Run

```bash
python modeling/train_models.py
```

Outputs
- `outputs/charts/roc_curve_models.png`
- `outputs/charts/feature_importance_models.png`
- `outputs/models/*.pkl`
- `outputs/models/cox_summary.csv` (if Cox model run)

Notes
- The script auto-detects a CSV in `outputs/` or `data/`. Ensure the dataset contains a binary target column named `default` or similar. If not detected, add a `default` column (0/1).
- The script uses XGBoost and LightGBM if installed; otherwise it will skip them.
