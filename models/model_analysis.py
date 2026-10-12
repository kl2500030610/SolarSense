"""
models/model_analysis.py
------------------------
Comprehensive model performance analysis for the Solar Power Plant dashboard.

This module complements the Regression and Ensemble pages by going deeper into
*how well* the models generalise. It covers:

  - Train / Validation / Test splitting  (60 / 20 / 20, stratified)
  - K-fold cross-validation (regression + classification)
  - Regression metrics        : RMSE, MAE, MSE, R², Adjusted R²
  - Classification metrics    : Accuracy, Precision, Recall, F1,
                                ROC-AUC, PR-AUC
        (the classification target is obtained by discretising AC_POWER into
         meaningful solar-power levels: Off / Low / High)
  - Probability calibration   : reliability diagram + Brier / log-loss
  - Hyperparameter search     : GridSearchCV, RandomizedSearchCV and a
        Bayesian optimiser (Gaussian-Process surrogate + Expected Improvement)

Results (metrics, tables) are returned as plain Python structures; plots are
written to static/analysis_plots/.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

try:  # optional dependency
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - older sklearn
    FrozenEstimator = None

from models.regression import TARGET, FEATURE_COLS, CAT_COLS

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "analysis_plots")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")

RANDOM_STATE = 42
MAX_SAMPLES = 30000      # cap rows used for train/val/test + CV (keeps it fast)
TUNE_SAMPLES = 5000      # smaller cap used for hyperparameter search
CV_FOLDS = 5             # k-fold cross-validation
TUNE_FOLDS = 3           # folds used during hyperparameter search
N_CLASSES = 3            # Off / Low / High


# ---------------------------------------------------------------------------
# Feature preparation & target discretisation
# ---------------------------------------------------------------------------

def prepare_features(df):
    """Return X (features) and y (continuous AC_POWER)."""
    cols_needed = [TARGET] + FEATURE_COLS + CAT_COLS
    subset = df[cols_needed].dropna().copy()
    if len(subset) > MAX_SAMPLES:
        subset = subset.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)
    X = subset[FEATURE_COLS + CAT_COLS].reset_index(drop=True)
    y = subset[TARGET].astype(float).reset_index(drop=True)
    return X, y, subset


def build_preprocessor():
    """Preprocessor: scale numeric features, one-hot encode PLANT_ID."""
    numeric = StandardScaler()
    categorical = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", numeric, FEATURE_COLS),
            ("cat", categorical, CAT_COLS),
        ],
        remainder="drop",
    )


def discretize_target(y, n_classes=N_CLASSES):
    """Discretise continuous AC_POWER into solar-power classes.

    Class 0 is always the "Off" class (AC_POWER == 0, i.e. night / no output).
    The remaining (n_classes - 1) classes split the non-zero values at equal
    quantiles, giving a balanced, physically meaningful classification target.

    Returns
    -------
    labels : np.ndarray[int]
    class_names : list[str]
    boundaries : list[float]
    """
    y = np.asarray(y, dtype=float)
    n_groups = n_classes - 1
    nonzero = y[y > 0]
    if nonzero.size == 0:
        raise ValueError("Cannot discretise target: all AC_POWER values are zero.")

    interior_pcts = np.linspace(0, 100, n_groups + 1)[1:-1]
    cuts = np.percentile(nonzero, interior_pcts) if interior_pcts.size else np.array([])

    labels = np.zeros(len(y), dtype=int)
    nz_idx = np.where(y > 0)[0]
    if cuts.size:
        groups = np.searchsorted(cuts, y[nz_idx], side="right")
        labels[nz_idx] = groups + 1
    else:
        labels[nz_idx] = 1

    boundaries = [0.0] + [float(c) for c in cuts] + [float(np.max(y))]

    if n_classes == 3:
        class_names = ["Off (0 W)", "Low", "High"]
    elif n_classes == 4:
        class_names = ["Off (0 W)", "Low", "Medium", "High"]
    else:
        class_names = ["Off (0 W)"] + [f"Level {i}" for i in range(1, n_classes)]

    return labels, class_names, boundaries


def _make_split(X, y_reg, y_clf):
    """60 / 20 / 20 stratified split into train / validation / test."""
    X_tmp, X_test, yr_tmp, yr_test, yc_tmp, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=RANDOM_STATE, stratify=y_clf
    )
    X_train, X_val, yr_train, yr_val, yc_train, yc_val = train_test_split(
        X_tmp, yr_tmp, yc_tmp, test_size=0.25, random_state=RANDOM_STATE, stratify=yc_tmp
    )
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "yr_train": yr_train, "yr_val": yr_val, "yr_test": yr_test,
        "yc_train": yc_train, "yc_val": yc_val, "yc_test": yc_test,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _regression_metrics(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else r2
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MSE": round(float(mean_squared_error(y_true, y_pred)), 4),
        "R2": round(float(r2), 4),
        "Adj_R2": round(float(adj_r2), 4),
    }


def _classification_metrics(y_true, y_pred, y_proba, n_classes):
    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }
    try:
        metrics["ROC_AUC"] = round(
            float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")), 4
        )
    except Exception:
        metrics["ROC_AUC"] = None
    try:
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])
        metrics["PR_AUC"] = round(
            float(average_precision_score(y_bin, y_proba, average="weighted")), 4
        )
    except Exception:
        metrics["PR_AUC"] = None
    return metrics


def _cv_regression(pipe, X, y, folds=CV_FOLDS):
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
    rmse = cross_val_score(pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
    mae = cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
    return {
        "R2_mean": round(float(np.mean(r2)), 4),
        "R2_std": round(float(np.std(r2)), 4),
        "RMSE_mean": round(float(-np.mean(rmse)), 4),
        "RMSE_std": round(float(np.std(rmse)), 4),
        "MAE_mean": round(float(-np.mean(mae)), 4),
        "MAE_std": round(float(np.std(mae)), 4),
    }


def _cv_classification(pipe, X, y, folds=CV_FOLDS):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)
    try:
        roc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc_ovr_weighted", n_jobs=-1)
    except Exception:
        roc = acc
    return {
        "Accuracy_mean": round(float(np.mean(acc)), 4),
        "Accuracy_std": round(float(np.std(acc)), 4),
        "F1_mean": round(float(np.mean(f1)), 4),
        "F1_std": round(float(np.std(f1)), 4),
        "ROC_AUC_mean": round(float(np.mean(roc)), 4),
        "ROC_AUC_std": round(float(np.std(roc)), 4),
    }


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def _grid_search(X, y):
    pipe = Pipeline([("pre", build_preprocessor()), ("model", RandomForestRegressor(
        min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE))])
    grid = {
        "model__n_estimators": [60, 120],
        "model__max_depth": [8, None],
    }
    search = GridSearchCV(pipe, grid, cv=TUNE_FOLDS, scoring="r2", n_jobs=-1)
    search.fit(X, y)
    order = np.argsort(search.cv_results_["rank_test_score"])
    top = [
        {
            "params": {k.replace("model__", ""): (v if v is not None else "None")
                       for k, v in search.cv_results_["params"][i].items()},
            "score": round(float(search.cv_results_["mean_test_score"][i]), 4),
        }
        for i in order[:5]
    ]
    return {
        "best_params": {k.replace("model__", ""): (v if v is not None else "None")
                        for k, v in search.best_params_.items()},
        "best_score": round(float(search.best_score_), 4),
        "n_candidates": len(search.cv_results_["params"]),
        "top": top,
    }


def _random_search(X, y):
    from scipy.stats import randint
    pipe = Pipeline([("pre", build_preprocessor()), ("model", RandomForestRegressor(
        min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE))])
    space = {
        "model__n_estimators": randint(40, 220),
        "model__max_depth": [6, 8, 10, 15, None],
        "model__min_samples_leaf": randint(1, 10),
    }
    search = RandomizedSearchCV(
        pipe, space, n_iter=10, cv=TUNE_FOLDS, scoring="r2",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    search.fit(X, y)
    return {
        "best_params": {k.replace("model__", ""): (v if v is not None else "None")
                        for k, v in search.best_params_.items()},
        "best_score": round(float(search.best_score_), 4),
        "n_candidates": 10,
    }


def _bayesian_search(X, y, n_iter=10):
    """Bayesian optimisation over a discrete grid of RandomForest hyperparameters.

    Uses a Gaussian-Process surrogate with a Matérn kernel and the Expected
    Improvement acquisition function (sklearn + scipy only, no extra deps).
    """
    n_est_grid = [50, 100, 150, 200]
    depth_grid = [6, 10, 15, None]
    combos = [(a, b) for a in n_est_grid for b in depth_grid]

    def encode(combo):
        n_est, depth = combo
        return [float(n_est), float(depth if depth is not None else 30)]

    def evaluate(combo):
        n_est, depth = combo
        pipe = Pipeline([("pre", build_preprocessor()), ("model", RandomForestRegressor(
            n_estimators=n_est, max_depth=depth, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE))])
        return float(cross_val_score(pipe, X, y, cv=TUNE_FOLDS, scoring="r2", n_jobs=-1).mean())

    rng = np.random.RandomState(RANDOM_STATE)
    init_idx = list(rng.choice(len(combos), size=min(4, len(combos)), replace=False))

    observed_x, observed_y, history = [], [], []
    for i in init_idx:
        score = evaluate(combos[i])
        observed_x.append(encode(combos[i]))
        observed_y.append(score)
        history.append({"iteration": len(history) + 1, "params": combos[i], "score": round(score, 4)})

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0], nu=2.5)
    while len(history) < n_iter:
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=RANDOM_STATE
        )
        gp.fit(np.array(observed_x), np.array(observed_y))

        tried = {tuple(x) for x in observed_x}
        remaining = [c for c in combos if tuple(encode(c)) not in tried]
        if not remaining:
            break

        pool = np.array([encode(c) for c in remaining])
        mu, sigma = gp.predict(pool, return_std=True)
        best = max(observed_y)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu - best) / sigma
            ei = (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma <= 1e-9] = 0.0
        pick = remaining[int(np.argmax(ei))]

        score = evaluate(pick)
        observed_x.append(encode(pick))
        observed_y.append(score)
        history.append({"iteration": len(history) + 1, "params": pick, "score": round(score, 4)})

    best_entry = max(history, key=lambda h: h["score"])
    best_n, best_depth = best_entry["params"]
    return {
        "best_params": {
            "n_estimators": best_n,
            "max_depth": best_depth if best_depth is not None else "None",
        },
        "best_score": round(float(best_entry["score"]), 4),
        "n_candidates": len(history),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_regression_comparison(results):
    names = list(results.keys())
    metrics = ["R2", "RMSE", "MAE"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#c96f11", "#f2b134", "#8a4b08", "#d99a4e"]
    for ax, metric in zip(axes, metrics):
        vals = [results[n]["test_metrics"][metric] for n in names]
        bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="white")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Regression — Model Comparison (Test Set)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = "regression_comparison.png"
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_pred_vs_actual(y_true, y_pred, name):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.15, s=8, color="#c96f11", edgecolors="none")
    lims = [0, max(np.max(y_true), np.max(y_pred)) * 1.05]
    ax.plot(lims, lims, "--", color="#333", linewidth=1)
    ax.set_xlabel("Actual AC_POWER")
    ax.set_ylabel("Predicted AC_POWER")
    ax.set_title(f"Actual vs Predicted — {name} (Test Set)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    fname = "regression_pred_vs_actual.png"
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120)
    plt.close(fig)
    return fname


def _plot_classification_comparison(results):
    names = list(results.keys())
    metrics = ["Accuracy", "F1", "ROC_AUC"]
    labels = ["Accuracy", "F1 (weighted)", "ROC-AUC"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#c96f11", "#f2b134", "#8a4b08", "#d99a4e"]
    for ax, metric, lab in zip(axes, metrics, labels):
        vals = [results[n]["test_metrics"][metric] or 0 for n in names]
        bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="white")
        ax.set_title(lab, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Classification — Model Comparison (Test Set)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = "classification_comparison.png"
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_confusion_matrix(cm, class_names, name):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name} (Test Set)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fname = "confusion_matrix.png"
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120)
    plt.close(fig)
    return fname


def _plot_calibration(y_true, proba_uncal, proba_cal, name):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")

    conf_u = proba_uncal.max(axis=1)
    correct_u = (proba_uncal.argmax(axis=1) == y_true).astype(int)
    pt_u, pp_u = calibration_curve(correct_u, conf_u, n_bins=10, strategy="quantile")
    ax.plot(pp_u, pt_u, "o-", color="#8a4b08", label="Uncalibrated")

    if proba_cal is not None:
        conf_c = proba_cal.max(axis=1)
        correct_c = (proba_cal.argmax(axis=1) == y_true).astype(int)
        pt_c, pp_c = calibration_curve(correct_c, conf_c, n_bins=10, strategy="quantile")
        ax.plot(pp_c, pt_c, "s-", color="#c96f11", label="Calibrated (isotonic)")

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives (accuracy)")
    ax.set_title(f"Calibration / Reliability — {name}")
    ax.legend(loc="best")
    fig.tight_layout()
    fname = "calibration_curve.png"
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120)
    plt.close(fig)
    return fname


def _plot_hyperparameter_comparison(hp):
    names = ["Grid Search", "Random Search", "Bayesian Opt."]
    keys = ["grid", "random", "bayesian"]
    scores = [hp[k]["best_score"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, scores, color=["#c96f11", "#f2b134", "#8a4b08"], edgecolor="white")
    ax.set_ylabel("Best CV R²")
    ax.set_title("Hyperparameter Search — Best Cross-Validated R²")
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fname = "hyperparameter_comparison.png"
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120)
    plt.close(fig)
    return fname


def _plot_bayesian_convergence(history):
    iters = [h["iteration"] for h in history]
    scores = [h["score"] for h in history]
    best_so_far = np.maximum.accumulate(scores)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, scores, "o-", color="#f2b134", label="Observed R²")
    ax.plot(iters, best_so_far, "s-", color="#8a4b08", label="Best so far")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("CV R²")
    ax.set_title("Bayesian Optimisation — Convergence")
    ax.legend(loc="best")
    fig.tight_layout()
    fname = "bayesian_convergence.png"
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_analysis(df):
    """Run the complete model-performance analysis and return a results dict."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y_reg, subset = prepare_features(df)
    y_clf, class_names, boundaries = discretize_target(y_reg, N_CLASSES)

    split = _make_split(X, y_reg, y_clf)
    X_train, X_val, X_test = split["X_train"], split["X_val"], split["X_test"]
    yr_train, yr_test = split["yr_train"], split["yr_test"]
    yc_train, yc_test = split["yc_train"], split["yc_test"]
    n_features = X_train.shape[1]

    feature_matrix = build_preprocessor()

    # --- Regression models -------------------------------------------------
    reg_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE),
    }
    if XGBOOST_AVAILABLE:
        reg_models["XGBoost"] = XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)

    regression = {"models": {}, "best_model": None}
    reg_pipelines = {}
    for name, model in reg_models.items():
        pipe = Pipeline([("pre", feature_matrix), ("model", model)])
        pipe.fit(X_train, yr_train)
        y_pred = pipe.predict(X_test)
        regression["models"][name] = {
            "test_metrics": _regression_metrics(yr_test, y_pred, n_features),
            "cv": _cv_regression(pipe, X_train, yr_train),
        }
        reg_pipelines[name] = pipe
    regression["best_model"] = max(
        regression["models"], key=lambda m: regression["models"][m]["test_metrics"]["R2"])

    # --- Classification models --------------------------------------------
    clf_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE),
    }
    if XGBOOST_AVAILABLE:
        clf_models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE,
            verbosity=0, eval_metric="mlogloss")

    classification = {"models": {}, "best_model": None, "class_names": class_names}
    clf_pipelines, clf_proba = {}, {}
    for name, model in clf_models.items():
        pipe = Pipeline([("pre", feature_matrix), ("model", model)])
        pipe.fit(X_train, yc_train)
        y_pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)
        classification["models"][name] = {
            "test_metrics": _classification_metrics(yc_test, y_pred, proba, N_CLASSES),
            "cv": _cv_classification(pipe, X_train, yc_train),
        }
        clf_pipelines[name] = pipe
        clf_proba[name] = proba
    best_clf = max(
        classification["models"], key=lambda m: classification["models"][m]["test_metrics"]["F1"])
    classification["best_model"] = best_clf
    best_pipe = clf_pipelines[best_clf]
    best_pred = best_pipe.predict(X_test)
    cm = confusion_matrix(yc_test, best_pred, labels=np.arange(N_CLASSES))
    classification["confusion_matrix"] = cm.tolist()

    # --- Calibration of the best classifier -------------------------------
    calibration = {"method": "uncalibrated", "brier": None, "log_loss": None,
                   "calibrated_brier": None, "calibrated_log_loss": None}
    proba_uncal = clf_proba[best_clf]
    calibration["brier"] = round(float(brier_score_loss(
        (best_pipe.predict(X_test) == yc_test).astype(int),
        proba_uncal.max(axis=1))), 4)
    calibration["log_loss"] = round(float(log_loss(yc_test, proba_uncal, labels=np.arange(N_CLASSES))), 4)

    proba_cal = None
    if FrozenEstimator is not None:
        try:
            calibrated = CalibratedClassifierCV(
                FrozenEstimator(best_pipe), method="isotonic")
            calibrated.fit(X_val, split["yc_val"])
            proba_cal = calibrated.predict_proba(X_test)
            calibration["method"] = "isotonic (fitted on validation set)"
            calibration["calibrated_brier"] = round(float(brier_score_loss(
                (proba_cal.argmax(axis=1) == yc_test).astype(int), proba_cal.max(axis=1))), 4)
            calibration["calibrated_log_loss"] = round(float(
                log_loss(yc_test, proba_cal, labels=np.arange(N_CLASSES))), 4)
        except Exception:
            proba_cal = None

    # --- Hyperparameter search (on a subsample, RandomForest) -------------
    tune_n = min(TUNE_SAMPLES, len(X_train))
    X_tune = X_train.sample(n=tune_n, random_state=RANDOM_STATE)
    yr_tune = yr_train.loc[X_tune.index]
    hp = {
        "grid": _grid_search(X_tune, yr_tune),
        "random": _random_search(X_tune, yr_tune),
        "bayesian": _bayesian_search(X_tune, yr_tune, n_iter=10),
    }

    # --- Plots ------------------------------------------------------------
    plots = []
    plots.append({"file": _plot_regression_comparison(regression["models"]),
                  "title": "Regression Model Comparison"})
    best_reg = regression["best_model"]
    plots.append({"file": _plot_pred_vs_actual(
        yr_test.values, reg_pipelines[best_reg].predict(X_test), best_reg),
        "title": f"Actual vs Predicted — {best_reg}"})
    plots.append({"file": _plot_classification_comparison(classification["models"]),
                  "title": "Classification Model Comparison"})
    plots.append({"file": _plot_confusion_matrix(np.array(cm), class_names, best_clf),
                  "title": f"Confusion Matrix — {best_clf}"})
    plots.append({"file": _plot_calibration(yc_test, proba_uncal, proba_cal, best_clf),
                  "title": f"Calibration — {best_clf}"})
    plots.append({"file": _plot_hyperparameter_comparison(hp),
                  "title": "Hyperparameter Search Comparison"})
    plots.append({"file": _plot_bayesian_convergence(hp["bayesian"]["history"]),
                  "title": "Bayesian Optimisation Convergence"})

    # --- Assemble results -------------------------------------------------
    counts = np.bincount(y_clf, minlength=N_CLASSES)
    results = {
        "target": TARGET,
        "n_classes": N_CLASSES,
        "class_names": class_names,
        "class_counts": counts.tolist(),
        "class_pct": [round(100.0 * c / len(y_clf), 2) for c in counts],
        "boundaries": [round(b, 2) for b in boundaries],
        "n_samples": len(subset),
        "split": {
            "train": len(X_train), "val": len(X_val), "test": len(X_test),
        },
        "regression": regression,
        "classification": classification,
        "calibration": calibration,
        "hyperparameter": hp,
        "plots": plots,
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    results["findings"] = _generate_findings(results)
    return results


def _generate_findings(results):
    """Build human-readable findings from the analysis results."""
    f = []
    reg = results["regression"]
    clf = results["classification"]
    best_r = reg["best_model"]
    best_c = clf["best_model"]
    r2 = reg["models"][best_r]["test_metrics"]["R2"]
    rmse = reg["models"][best_r]["test_metrics"]["RMSE"]
    acc = clf["models"][best_c]["test_metrics"]["Accuracy"]
    f1 = clf["models"][best_c]["test_metrics"]["F1"]

    f.append(
        f"Data split: <strong>{results['split']['train']:,}</strong> train / "
        f"<strong>{results['split']['val']:,}</strong> validation / "
        f"<strong>{results['split']['test']:,}</strong> test samples (60/20/20, stratified)."
    )
    f.append(
        f"Regression: best model is <strong>{best_r}</strong> with test "
        f"R² = {r2:.4f} and RMSE = {rmse:.4f}."
    )
    f.append(
        f"Classification: AC_POWER was discretised into "
        f"<strong>{', '.join(results['class_names'])}</strong>. Best model is "
        f"<strong>{best_c}</strong> with accuracy = {acc:.4f} and weighted F1 = {f1:.4f}."
    )

    cal = results["calibration"]
    if cal.get("calibrated_brier") is not None:
        direction = "improved" if cal["calibrated_brier"] <= cal["brier"] else "did not improve"
        f.append(
            f"Calibration: Brier score {cal['brier']:.4f} (uncalibrated) → "
            f"{cal['calibrated_brier']:.4f} (isotonic), which <strong>{direction}</strong> "
            f"the reliability of the predicted probabilities."
        )
    else:
        f.append(
            f"Calibration: Brier score = {cal['brier']:.4f}, log-loss = "
            f"{cal['log_loss']:.4f} (calibrated probabilities unavailable in this environment)."
        )

    hp = results["hyperparameter"]
    hw = max(hp, key=lambda k: hp[k]["best_score"])
    label = {"grid": "Grid Search", "random": "Random Search", "bayesian": "Bayesian Optimisation"}[hw]
    f.append(
        f"Hyperparameter search: <strong>{label}</strong> produced the best cross-validated "
        f"RandomForest R² = {hp[hw]['best_score']:.4f} with parameters "
        f"<code>{hp[hw]['best_params']}</code>."
    )

    if not results["xgboost_available"]:
        f.append("<strong>Note:</strong> XGBoost is not installed, so it was excluded from the comparison.")

    f.append(
        "Cross-validation on the training folds produces mean ± std scores; a small "
        "standard deviation indicates the models generalise consistently across folds."
    )
    return f


# Alias kept for symmetry with the other modules.
run_full_model_analysis = run_full_analysis
