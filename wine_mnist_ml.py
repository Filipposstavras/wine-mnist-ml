#!/usr/bin/env python3
"""
Wine Quality — Red (Regression)
End-to-end ML pipeline for predicting wine quality from physicochemical features.

Steps:
1) Download UCI dataset (zip) and load red wine CSV
2) Describe data: types, ranges, completeness
3) Histograms for all features (+ skewness); hints for Gaussianization
4) Correlation with quality; simple heatmap
5) Stratified 80/20 split by binned quality
6) Pipeline: StandardScaler + LinearRegression; metrics on test set
7) Actual vs Predicted plot
8) 10-fold cross validation on R^2

Run:
    python wine_quality_red.py
Optional args:
    --test_size 0.2 --random_state 42 --tolerance 0.5
"""

import argparse
import io
import json
import math
import os
import sys
import zipfile
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
DATA_DIR = "data"
REPORTS_DIR = "reports"
FIGURES_DIR = "figures"


def safe_makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def download_and_load_red_wine(data_dir=DATA_DIR) -> pd.DataFrame:
    """Download UCI wine quality dataset zip and return red wine dataframe."""
    safe_makedirs(data_dir)
    zip_path = os.path.join(data_dir, "wine_quality.zip")
    red_csv_path = os.path.join(data_dir, "winequality-red.csv")

    if not os.path.exists(red_csv_path):
        print("Downloading UCI wine quality dataset ...")
        # Some servers block default user-agents; set a simple header
        req = Request(UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as resp:
            zip_bytes = resp.read()
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        print(f"Saved zip to {zip_path}")

        print("Extracting files ...")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Extract only the red wine CSV
            members = [m for m in zf.namelist() if m.endswith("winequality-red.csv")]
            if not members:
                # Extract all and hope filename variant exists
                zf.extractall(data_dir)
            else:
                zf.extract(members[0], data_dir)

        # After extraction, ensure file is at expected path
        # Search if nested folder was created
        if not os.path.exists(red_csv_path):
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f == "winequality-red.csv":
                        src = os.path.join(root, f)
                        os.replace(src, red_csv_path)
                        break

    print(f"Loading red wine CSV from {red_csv_path}")
    df = pd.read_csv(red_csv_path, sep=";")
    return df


def descriptive_stats(df: pd.DataFrame):
    """Compute and save descriptive statistics and completeness info."""
    safe_makedirs(REPORTS_DIR)

    dtypes = df.dtypes.astype(str)
    completeness = df.isna().sum().rename("missing_count")
    describe = df.describe(include="all")

    dtypes.to_csv(os.path.join(REPORTS_DIR, "dtypes.csv"))
    completeness.to_csv(os.path.join(REPORTS_DIR, "missing_counts.csv"))
    describe.to_csv(os.path.join(REPORTS_DIR, "describe.csv"))

    print("\n=== Data Types ===")
    print(dtypes)
    print("\n=== Missing values per column ===")
    print(completeness)
    print("\n=== Describe() ===")
    print(describe)


def plot_histograms(df: pd.DataFrame):
    """Plot histograms for all numeric features and save figures. Also compute skewness."""
    safe_makedirs(FIGURES_DIR)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "quality"]
    n_cols = 3
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    plt.figure(figsize=(12, 4 * n_rows))

    skew_info = {}
    for i, col in enumerate(numeric_cols, start=1):
        plt.subplot(n_rows, n_cols, i)
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f"{col}")
        plt.xlabel(col)
        plt.ylabel("count")
        skew = df[col].skew()
        skew_info[col] = float(skew)

    plt.tight_layout()
    grid_path = os.path.join(FIGURES_DIR, "histograms_grid.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"Saved histogram grid to {grid_path}")

    with open(os.path.join(REPORTS_DIR, "skewness.json"), "w") as f:
        json.dump(skew_info, f, indent=2)

    # Print transformation hints
    print("\n=== Skewness (|skew| > 1 suggests considering transforms like log1p) ===")
    for col, sk in sorted(skew_info.items(), key=lambda x: -abs(x[1])):
        print(f"{col:20s} skew={sk: .3f}")


def correlation_with_quality(df: pd.DataFrame):
    """Compute and save correlations with quality and a simple heatmap."""
    safe_makedirs(REPORTS_DIR)
    safe_makedirs(FIGURES_DIR)

    corr = df.corr(numeric_only=True)
    corr.to_csv(os.path.join(REPORTS_DIR, "correlation_matrix.csv"))

    if "quality" not in corr.columns:
        print("Warning: 'quality' not found in correlation matrix columns.")
        return

    target_corr = corr["quality"].drop(labels=["quality"])
    target_corr.sort_values(ascending=False).to_csv(os.path.join(REPORTS_DIR, "corr_with_quality.csv"))
    print("\n=== Correlation with quality (descending) ===")
    print(target_corr.sort_values(ascending=False))

    # Simple heatmap using matplotlib
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    heatmap_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved correlation heatmap to {heatmap_path}")


def stratified_split(df: pd.DataFrame, test_size: float, random_state: int):
    """Stratify by binned quality to preserve distribution."""
    bins = [0, 4.5, 5.5, 6.5, 10]
    labels = [4, 5, 6, 7]
    df = df.copy()
    df["quality_bin"] = pd.cut(df["quality"], bins=bins, labels=labels, include_lowest=True)

    X = df.drop(columns=["quality", "quality_bin"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=df["quality_bin"]
    )
    return X_train, X_test, y_train, y_test, X, y


def train_and_evaluate(X_train, X_test, y_train, y_test, tolerance: float):
    """Train Linear Regression in a StandardScaler pipeline and compute metrics."""
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # MAPE: safe division in case of zeros, though wine quality >= 3 typically
    mape = (np.abs((y_test - y_pred) / y_test)).mean() * 100
    acc_tol = (np.abs(y_test - y_pred) <= tolerance).mean()

    metrics = {
        "R2": float(r2),
        "MAE": float(mae),
        "MSE": float(mse),
        "MAPE_percent": float(mape),
        "Accuracy_within_tolerance": float(acc_tol),
        "Tolerance": float(tolerance),
    }
    return pipe, y_pred, metrics


def plot_actual_vs_pred(y_test, y_pred):
    safe_makedirs(FIGURES_DIR)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
    plt.plot(lims, lims, linewidth=2)
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title("Actual vs Predicted — Red Wine Quality")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "actual_vs_predicted.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Actual vs Predicted plot to {out_path}")


def cross_validation_r2(X, y, random_state: int):
    """10-fold cross validation on R^2."""
    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    scores = cross_val_score(pipe, X, y, scoring="r2", cv=cv)
    return float(scores.mean()), float(scores.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.5, help="Accuracy tolerance |y-ŷ| ≤ tolerance")
    args = parser.parse_args()

    safe_makedirs(REPORTS_DIR)
    safe_makedirs(FIGURES_DIR)

    # 1) Data
    df = download_and_load_red_wine(DATA_DIR)

    # 2) Features
    features = [c for c in df.columns if c != "quality"]
    with open(os.path.join(REPORTS_DIR, "features.json"), "w") as f:
        json.dump(features, f, indent=2)
    print("\n=== Features (predictors) ===")
    print(features)

    # 3) Descriptive stats
    descriptive_stats(df)

    # 4) Histograms & skewness
    plot_histograms(df)

    # 5) Correlations
    correlation_with_quality(df)

    # 6) Stratified split
    X_train, X_test, y_train, y_test, X_all, y_all = stratified_split(df, args.test_size, args.random_state)

    # 7) Train & evaluate
    model, y_pred, metrics = train_and_evaluate(X_train, X_test, y_train, y_test, args.tolerance)
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 8) Plot actual vs predicted
    plot_actual_vs_pred(y_test.values, y_pred)

    # 9) 10-fold CV
    cv_mean, cv_std = cross_validation_r2(X_all, y_all, args.random_state)
    cv_report = {"cv_r2_mean": cv_mean, "cv_r2_std": cv_std}
    with open(os.path.join(REPORTS_DIR, "cv_r2.json"), "w") as f:
        json.dump(cv_report, f, indent=2)
    print(f"\n=== 10-fold CV R^2 ===\nmean={cv_mean:.4f}  std={cv_std:.4f}")

    # 10) Check if test R^2 within CV interval (mean ± std)
    lower, upper = cv_mean - cv_std, cv_mean + cv_std
    within = lower <= metrics["R2"] <= upper
    print(f"\nIs test R^2 within CV mean±std? {within}  (interval: [{lower:.4f}, {upper:.4f}])")
    with open(os.path.join(REPORTS_DIR, "cv_vs_test.json"), "w") as f:
        json.dump({"cv_interval": [lower, upper], "test_r2": metrics["R2"], "within": within}, f, indent=2)

    print("\nDone. Reports saved to ./reports and plots to ./figures\n")


if __name__ == "__main__":
    main()
