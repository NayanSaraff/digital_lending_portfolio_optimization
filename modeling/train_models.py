"""Train classification and survival models on project data.

Usage:
  python modeling/train_models.py

Outputs:
 - outputs/models/*.pkl (trained models)
 - outputs/charts/roc_curve_models.png
 - outputs/charts/feature_importance_models.png

This script attempts to auto-detect a suitable input CSV in outputs/ or data/.
"""
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from lifelines import CoxPHFitter
except Exception:
    CoxPHFitter = None


ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT / 'outputs' / 'merged_stage4.csv',
    ROOT / 'outputs' / 'risk_scored_portfolio.csv',
    ROOT / 'data' / 'risk_scored_portfolio.csv',
    ROOT / 'data' / 'loan_outcome.csv',
    ROOT / 'outputs' / 'merged.csv',
]

OUT_CHARTS = ROOT / 'outputs' / 'charts'
OUT_MODELS = ROOT / 'outputs' / 'models'
OUT_CHARTS.mkdir(parents=True, exist_ok=True)
OUT_MODELS.mkdir(parents=True, exist_ok=True)


def find_input_file():
    for p in DATA_CANDIDATES:
        if p.exists():
            print('Using dataset:', p)
            return p
    # fallback: search csv files in outputs
    for p in (ROOT / 'outputs').glob('*.csv'):
        print('Found candidate:', p)
        return p
    raise FileNotFoundError('No input CSV found in expected locations.')


def load_data(path: Path):
    df = pd.read_csv(path)
    print('Loaded', len(df), 'rows and', len(df.columns), 'columns')
    return df

def detect_target(df):
    candidates = ['default', 'is_default', 'loan_default', 'default_flag', 'defaulted', 'loan_status']
    for c in candidates:
        if c in df.columns:
            print('Detected target column:', c)
            # normalize to binary 0/1
            y = df[c]
            if y.dtype == 'O':
                y = y.replace({'default': 1, 'nondefault': 0, 'paid': 0, 'charged_off': 1})
            return c
    # try columns with binary values
    for c in df.columns:
        vals = df[c].dropna().unique()
        if set(vals).issubset({0,1}):
            print('Detected binary-looking target column:', c)
            return c
    raise ValueError('No target column detected. Add a binary target (e.g., "default") to dataset.')


def preprocess(df, target_col):
    df = df.copy()
    # Drop identifiers and text-heavy columns
    drop_like = ['id', 'loan_id', 'customer_id', 'application_id']
    for d in drop_like:
        if d in df.columns:
            df = df.drop(columns=[d])
    # Keep numeric and categorical
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Simple imputation for numeric
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Encode categorical using one-hot (limited)
    if cat_cols:
        X_cat = pd.get_dummies(X[cat_cols].fillna('MISSING'), drop_first=True)
    else:
        X_cat = pd.DataFrame(index=X.index)

    X_num = X[num_cols].copy()
    imp = SimpleImputer(strategy='median')
    if not X_num.empty:
        X_num[:] = imp.fit_transform(X_num)

    X_processed = pd.concat([X_num, X_cat], axis=1)
    print('Processed features shape:', X_processed.shape)
    return X_processed, y


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    scaler = StandardScaler()
    if X_train.shape[1] > 0:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s = X_train
        X_test_s = X_test

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    yhat = lr.predict_proba(X_test_s)[:, 1]
    results['LogisticRegression'] = roc_auc_score(y_test, yhat)
    with open(OUT_MODELS / 'logistic.pkl', 'wb') as f:
        pickle.dump(lr, f)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    yhat = rf.predict_proba(X_test)[:, 1]
    results['RandomForest'] = roc_auc_score(y_test, yhat)
    with open(OUT_MODELS / 'random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)

    # XGBoost
    if xgb is not None:
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_clf.fit(X_train, y_train)
        yhat = xgb_clf.predict_proba(X_test)[:, 1]
        results['XGBoost'] = roc_auc_score(y_test, yhat)
        with open(OUT_MODELS / 'xgboost.pkl', 'wb') as f:
            pickle.dump(xgb_clf, f)
    else:
        print('xgboost not available; skipping')

    # LightGBM
    if lgb is not None:
        lgb_clf = lgb.LGBMClassifier(random_state=42)
        lgb_clf.fit(X_train, y_train)
        yhat = lgb_clf.predict_proba(X_test)[:, 1]
        results['LightGBM'] = roc_auc_score(y_test, yhat)
        with open(OUT_MODELS / 'lightgbm.pkl', 'wb') as f:
            pickle.dump(lgb_clf, f)
    else:
        print('lightgbm not available; skipping')

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for name, model_auc in results.items():
        if name == 'LogisticRegression':
            probs = lr.predict_proba(X_test_s)[:, 1]
        elif name == 'RandomForest':
            probs = rf.predict_proba(X_test)[:, 1]
        elif name == 'XGBoost' and 'XGBoost' in results:
            probs = xgb_clf.predict_proba(X_test)[:, 1]
        elif name == 'LightGBM' and 'LightGBM' in results:
            probs = lgb_clf.predict_proba(X_test)[:, 1]
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={model_auc:.3f})")
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CHARTS / 'roc_curve_models.png')
    print('Saved ROC plot to', OUT_CHARTS / 'roc_curve_models.png')

    # Feature importance (RF and tree-based)
    fi = None
    if hasattr(rf, 'feature_importances_'):
        fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(30)
    elif 'XGBoost' in results:
        fi = pd.Series(xgb_clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(30)
    elif 'LightGBM' in results:
        fi = pd.Series(lgb_clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(30)

    if fi is not None:
        plt.figure(figsize=(8, 10))
        sns.barplot(x=fi.values, y=fi.index)
        plt.title('Top feature importances')
        plt.tight_layout()
        plt.savefig(OUT_CHARTS / 'feature_importance_models.png')
        print('Saved feature importance to', OUT_CHARTS / 'feature_importance_models.png')

    print('AUC results:')
    for k, v in results.items():
        print(f' - {k}: {v:.4f}')


def try_cox(df):
    if CoxPHFitter is None:
        print('lifelines not installed; skipping CoxPH model')
        return
    # Need duration and event columns
    dur_cands = ['time_to_default', 'time_to_event', 'duration_months', 'tenure']
    evt_cands = ['default', 'is_default', 'event']
    dur_col = next((c for c in dur_cands if c in df.columns), None)
    evt_col = next((c for c in evt_cands if c in df.columns), None)
    if dur_col and evt_col:
        df_cox = df[[dur_col, evt_col]].join(df.select_dtypes(include=[np.number]).drop(columns=[dur_col], errors='ignore')).dropna()
        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col=dur_col, event_col=evt_col)
        print('CoxPH summary:')
        print(cph.summary.head())
        # Save coefficients
        cph.summary.to_csv(OUT_MODELS / 'cox_summary.csv')
    else:
        print('Duration/event columns not found; skipping CoxPH')


def main():
    inp = find_input_file()
    df = load_data(inp)
    try:
        target = detect_target(df)
    except ValueError as e:
        print(e)
        sys.exit(1)
    X, y = preprocess(df, target)
    train_and_evaluate(X, y)
    try_cox(df)


if __name__ == '__main__':
    main()
