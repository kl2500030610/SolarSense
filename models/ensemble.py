"""
models/ensemble.py
------------------
Ensemble models module for the Solar Power Plant EDA dashboard.

Target variable: AC_POWER (AC power output)
Features:
  - IRRADIATION (solar irradiation)
  - MODULE_TEMPERATURE (solar panel surface temperature)
  - AMBIENT_TEMPERATURE (air temperature)
  - PLANT_ID (categorical → one-hot encoded)

Models:
  - RandomForestRegressor
  - XGBRegressor (if xgboost available)
  - DecisionTreeRegressor
  - AdaBoostRegressor
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "ensemble_plots")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models", "ensemble_pipeline.pkl")

TARGET = "AC_POWER"
FEATURE_COLS = ["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"]
CAT_COLS = ["PLANT_ID"]

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def prepare_features(df):
    """Return X (features) and y (target) ready for modelling."""
    cols_needed = [TARGET] + FEATURE_COLS + CAT_COLS
    subset = df[cols_needed].dropna().copy()
    subset[CAT_COLS] = subset[CAT_COLS].astype(str)
    X = subset[FEATURE_COLS + CAT_COLS]
    y = subset[TARGET]
    return X, y


def build_pipeline():
    """Build a preprocessor + model pipeline."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"]),
            ("cat", categorical_transformer, ["PLANT_ID"]),
        ],
        remainder="drop",
    )
    return preprocessor


def _eval(y_true, y_pred, n_features):
    """Compute regression metrics."""
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else r2
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MSE": round(float(mean_squared_error(y_true, y_pred)), 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Adj_R2": round(adj_r2, 4),
    }


def train_and_evaluate(df):
    """Train all ensemble models, evaluate, generate plots and return full results dict."""
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_pipeline()

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=5,
            n_jobs=-1, random_state=42,
        ),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=15, min_samples_leaf=5, random_state=42,
        ),
        "AdaBoost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=5, random_state=42),
            n_estimators=100, learning_rate=0.1, random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0,
        )

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    results = {
        "target": TARGET,
        "features": FEATURE_COLS + [f"PLANT_ID ({c})" for c in sorted(df[CAT_COLS[0]].astype(str).unique())[1:]],
        "n_features": X_train.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "models": {},
        "plots": [],
        "best_model": None,
        "feature_importances": None,
        "xgboost_available": XGBOOST_AVAILABLE,
    }

    trained_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = _eval(y_test, y_pred, X_train.shape[1])
        trained_pipelines[name] = pipe

        results["models"][name] = {
            "metrics": metrics,
            "y_pred": y_pred.tolist(),
        }

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test.values, y_pred, alpha=0.15, s=8, color="#c96f11", edgecolors="none")
        lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
        ax.plot(lims, lims, "--", color="#333", linewidth=1)
        ax.set_xlabel("Actual AC_POWER")
        ax.set_ylabel("Predicted AC_POWER")
        ax.set_title(f"Actual vs Predicted — {name}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        fig.tight_layout()
        fname = f"actual_vs_predicted_{name.lower().replace(' ', '_')}.png"
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120)
        plt.close(fig)
        results["plots"].append({"file": fname, "title": f"Actual vs Predicted — {name}"})

        residuals = y_test.values - y_pred
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(residuals, bins=80, color="#c96f11", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="#333", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Residuals Distribution — {name}")
        fig.tight_layout()
        fname = f"residuals_{name.lower().replace(' ', '_')}.png"
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120)
        plt.close(fig)
        results["plots"].append({"file": fname, "title": f"Residuals — {name}"})

    model_names = list(results["models"].keys())
    metrics_to_plot = ["MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#c96f11", "#f2b134", "#8a4b08"]
    for idx, metric in enumerate(metrics_to_plot):
        vals = [results["models"][m]["metrics"][metric] for m in model_names]
        bars = axes[idx].bar(model_names, vals, color=colors, edgecolor="white")
        axes[idx].set_title(metric, fontsize=13, fontweight="bold")
        axes[idx].tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, vals):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                           f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = "model_comparison.png"
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    results["plots"].append({"file": fname, "title": "Model Comparison (MAE / RMSE / R²)"})

    best_name = max(results["models"], key=lambda m: results["models"][m]["metrics"]["R2"])
    best_pipe = trained_pipelines[best_name]
    results["best_model"] = best_name

    if hasattr(best_pipe.named_steps["regressor"], "feature_importances_"):
        importances = best_pipe.named_steps["regressor"].feature_importances_
        cat_feature_names = list(
            best_pipe.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .get_feature_names_out(["PLANT_ID"])
        )
        feature_names = ["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"] + cat_feature_names
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        feat_imp.plot.barh(ax=ax, color="#c96f11", edgecolor="white")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {best_name}")
        fig.tight_layout()
        fname = "feature_importance.png"
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120)
        plt.close(fig)
        results["plots"].append({"file": fname, "title": f"Feature Importance — {best_name}"})
        results["feature_importances"] = feat_imp.to_dict()

    joblib.dump(best_pipe, MODEL_PATH)

    results["findings"] = _generate_findings(results)

    return results


def _generate_findings(results):
    """Auto-generate human-readable findings from results."""
    findings = []
    best = results["best_model"]
    best_r2 = results["models"][best]["metrics"]["R2"]

    findings.append(
        f"Best model: <strong>{best}</strong> with R² = {results['models'][best]['metrics']['R2']:.4f}, "
        f"RMSE = {results['models'][best]['metrics']['RMSE']:.4f}, "
        f"MAE = {results['models'][best]['metrics']['MAE']:.4f}."
    )

    if best_r2 > 0.95:
        findings.append("The best model explains over 95% of the variance in AC power output — excellent fit.")
    elif best_r2 > 0.85:
        findings.append("The best model explains over 85% of the variance in AC power output — strong fit.")
    elif best_r2 > 0.70:
        findings.append("The best model explains over 70% of the variance in AC power output — good fit.")
    else:
        findings.append("The best model explains less than 70% of the variance — additional features may improve performance.")

    if results["feature_importances"]:
        top_feat = max(results["feature_importances"], key=results["feature_importances"].get)
        findings.append(
            f"<strong>{top_feat}</strong> is the most important feature according to the {best} model."
        )

    findings.append(
        "Approximately 49.5% of rows have AC_POWER = 0 (night-time). These zero-inflated records "
        "impact the linear models' R² but are realistic for a solar plant operating 24/7."
    )

    if not results["xgboost_available"]:
        findings.append(
            "<strong>Note:</strong> XGBoost is not installed. Install <code>xgboost</code> to include it in the ensemble."
        )

    findings.append(
        "The training set had {train:,} samples and the test set had {test:,} samples (80/20 split).".format(
            train=results["train_size"], test=results["test_size"]
        )
    )

    return findings


def predict_single(ambient_temp, module_temp, irradiation, plant_id):
    """Load the saved pipeline and predict AC_POWER for a single input."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No trained model found. Run the ensemble page first.")
    pipe = joblib.load(MODEL_PATH)
    X_new = pd.DataFrame([{
        "IRRADIATION": float(irradiation),
        "MODULE_TEMPERATURE": float(module_temp),
        "AMBIENT_TEMPERATURE": float(ambient_temp),
        "PLANT_ID": str(plant_id),
    }])
    pred = pipe.predict(X_new)[0]
    return max(0.0, round(float(pred), 4))


def get_plant_ids(df):
    """Return sorted unique plant IDs as strings."""
    return sorted(df["PLANT_ID"].astype(str).unique().tolist())


run_full_ensemble = train_and_evaluate